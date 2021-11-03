#!/bin/bash

if [ ! -f "/usr/local/lib/antlr-4.9-complete.jar" ];then
    echo "downloading antlr-4.9"
    curl -o /usr/local/lib/antlr-4.9-complete.jar https://www.antlr.org/download/antlr-4.9-complete.jar
    echo "download complete"
fi

java -Xmx500M -cp "/usr/local/lib/antlr-4.9-complete.jar:$CLASSPATH" org.antlr.v4.Tool -Dlanguage=Go -visitor -o . ../../internal/proxy/plan_parser/Plan.g4
