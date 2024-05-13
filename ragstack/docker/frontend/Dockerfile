FROM python:3.11-slim as builder
RUN pip install --no-cache-dir ragstack-ai-langflow
RUN pip show ragstack-ai-langflow | grep Location | cut -d ' ' -f 2 | xargs -I {} cp -r {}/langflow/frontend /tmp/frontend


FROM nginx
COPY --from=builder /tmp/frontend /usr/share/nginx/html
COPY /nginx.conf /etc/nginx/conf.d/default.conf
COPY start-nginx.sh /start-nginx.sh
RUN chmod +x /start-nginx.sh
CMD ["/start-nginx.sh"]
