#!/bin/bash

cd ./memberlist

go build -o main.so -buildmode=c-shared main.go

cd ..

cp ./memberlist/main.so ./memberlist.so
