# Natanyx Redrob Ranker Docker

Build and run locally:

```bash
docker build -t natanyx-redrob-ranker:latest .
docker run --rm -p 7860:7860 natanyx-redrob-ranker:latest
```

Open:

```text
http://localhost:7860
```

Push to Docker Hub:

```bash
docker tag natanyx-redrob-ranker:latest namanmani/natanyx-redrob-ranker:latest
docker push namanmani/natanyx-redrob-ranker:latest
```

Public run command after push:

```bash
docker run --rm -p 7860:7860 namanmani/natanyx-redrob-ranker:latest
```
