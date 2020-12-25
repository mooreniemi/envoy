use std::convert::Infallible;
use std::sync::Arc;
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
    println!("set up qa model config");
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
    config
}

fn qa_model(config: QuestionAnsweringConfig) -> QaModel {
    println!("set up qa model");
    let qa_model = QuestionAnsweringModel::new(config).expect("qa model failed to load");

    println!("return in wrapper");
    Arc::new(Mutex::new(qa_model))
}

async fn ask(qa_model: QaModel) -> Result<impl warp::Reply, Infallible> {
    println!("will ask");
    let question_1 = String::from("Where does Amy live ?");
    let context_1 = String::from("Amy lives in Amsterdam");
    let qa_input_1 = QaInput {
        question: question_1,
        context: context_1,
    };

    //    Get answer
    let answers = qa_model.lock().await.predict(&[qa_input_1], 1, 32);
    println!("{:?}", answers);
    let resp = answers[0][0].answer.clone();
    Ok(warp::reply::json(&resp))
}

#[tokio::main]
async fn main() {
    // NOTE: have to download the model before booting up
    let qa_model: QaModel = task::spawn_blocking(move || {
        println!("setting up qa model config");
        let c = qa_model_config();
        println!("finished setting up qa model config");

        println!("setting up qa model");
        let m = qa_model(c);
        println!("finished setting up qa model");
        m
    })
    .await
    .expect("got model");

    let ask_handler = warp::path!("ask")
        .map(move || qa_model.clone())
        .and_then(ask);

    warp::serve(ask_handler).run(([127, 0, 0, 1], 3030)).await;
}
