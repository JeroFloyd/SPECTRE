#!/bin/bash

URL=$1

echo "Testing /reset..."
curl -s -X POST $URL/reset > /dev/null && echo "PASSED" || echo "FAILED"

echo "Testing /step..."
curl -s -X POST $URL/step \
  -H "Content-Type: application/json" \
  -d '{"type":"primitive","name":"parse_data"}' > /dev/null && echo "PASSED" || echo "FAILED"

echo "Testing /state..."
curl -s $URL/state > /dev/null && echo "PASSED" || echo "FAILED"
