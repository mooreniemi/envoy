use serde_derive::Deserialize;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tokio::task;
use warp::Filter;

use rust_bert::bert::{BertConfigResources, BertModelResources, BertVocabResources};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::question_answering::{
    QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
};
use rust_bert::resources::{RemoteResource, Resource};

type QaModel = Arc<Mutex<QuestionAnsweringModel>>;

fn qa_model_config() -> QuestionAnsweringConfig {
    let start = Instant::now();
    let config = QuestionAnsweringConfig::new(
        ModelType::Bert,
        Resource::Remote(RemoteResource::from_pretrained(BertModelResources::BERT_QA)),
        Resource::Remote(RemoteResource::from_pretrained(
            BertConfigResources::BERT_QA,
        )),
        Resource::Remote(RemoteResource::from_pretrained(BertVocabResources::BERT_QA)),
        None,  //merges resource only relevant with ModelType::Roberta
        false, //lowercase
        false,
        None,
    );
    log::debug!("set up qa model config took {:?}", start.elapsed());
    config
}

fn qa_model(config: QuestionAnsweringConfig) -> QaModel {
    let start = Instant::now();
    let qa_model = QuestionAnsweringModel::new(config).expect("qa model failed to load");
    log::debug!("set up qa model took {:?}", start.elapsed());
    // NOTE: because *mut torch_sys::C_tensor cannot be shared between threads safely
    Arc::new(Mutex::new(qa_model))
}

async fn ask(query: QaQuery, qa_model: QaModel) -> Result<impl warp::Reply, Infallible> {
    let qa_input = QaInput {
        question: query.question.clone(),
        context: query.context.clone(),
    };
    log::info!(
        "ask for context={:?} question={:?}",
        query.context.clone(),
        query.question.clone()
    );

    let start = Instant::now();
    let answers = qa_model.lock().await.predict(&[qa_input], 1, 32);
    let top_answer = answers[0][0].answer.clone();
    let top_score = answers[0][0].score.clone();
    log::info!(
        "top answer={:?} ({:?}) took {:?}",
        top_answer.clone(),
        top_score.clone(),
        start.elapsed()
    );

    let mut response = HashMap::new();
    response.insert("question", query.question);
    response.insert("context", query.context);
    response.insert("answer", top_answer);
    response.insert("score", top_score.to_string());

    Ok(warp::reply::json(&response))
}

#[derive(Debug, Deserialize)]
pub struct QaQuery {
    pub question: String,
    pub context: String,
}

fn with_model(
    qa: QaModel,
) -> impl Filter<Extract = (QaModel,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || qa.clone())
}

fn json_body() -> impl Filter<Extract = (QaQuery,), Error = warp::Rejection> + Clone {
    // When accepting a body, we want a JSON body
    // (and to reject huge payloads)...
    warp::body::content_length_limit(1024 * 16).and(warp::body::json())
}

#[tokio::main]
async fn main() {
    env_logger::init();

    // NOTE: have to download the model before booting up
    let qa_model: QaModel = task::spawn_blocking(move || {
        log::debug!("setting up qa model config");
        let c = qa_model_config();
        log::debug!("finished setting up qa model config");

        log::debug!("setting up qa model");
        let m = qa_model(c);
        log::debug!("finished setting up qa model");
        m
    })
    .await
    .expect("got model");

    let qp_handler = warp::path!("ask")
        .and(warp::get())
        .and(warp::query::<QaQuery>())
        .and(with_model(qa_model.clone()))
        .and_then(ask);

    let json_handler = warp::path!("ask")
        .and(warp::get())
        .and(json_body())
        .and(with_model(qa_model))
        .and_then(ask);

    warp::serve(qp_handler.or(json_handler))
        .run(([127, 0, 0, 1], 3030))
        .await;
}
