#!/bin/bash

if [ ! -f "/tmp/antlr-4.9-complete.jar" ];then
    curl -o "/tmp/antlr-4.9-complete.jar" https://www.antlr.org/download/antlr-4.9-complete.jar
fi

java -Xmx500M -cp "/tmp/antlr-4.9-complete.jar:$CLASSPATH" org.antlr.v4.Tool -Dlanguage=Go -visitor -o . ../../internal/proxy/plan_parser/Plan.g4
