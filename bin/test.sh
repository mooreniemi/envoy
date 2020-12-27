# minimal examples

echo "testing with query params"
curl -s "localhost:3030/ask?question=where%20does%20amy%20live&contexts=amy%20lives%20at%20home" | jq

echo "testing with json"
curl -s --header "Content-Type: application/json" --request GET --data '{"question": "who is tesla", "contexts": ["tesla was an inventor who competed with thomas edison.", "tesla is a car company known for electric vehicles.", "tesla is run by elon musk."]}' localhost:3030/ask | jq
