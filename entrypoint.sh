#!/bin/bash
service nginx start
cd app
uwsgi --ini /app/uwsgi.ini