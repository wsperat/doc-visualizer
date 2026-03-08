FROM lfoppiano/grobid:0.8.0

EXPOSE 8070

HEALTHCHECK --interval=15s --timeout=5s --retries=20 CMD bash -lc 'exec 3<>/dev/tcp/127.0.0.1/8070'
