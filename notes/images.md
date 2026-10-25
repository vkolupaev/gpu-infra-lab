# Lab images

Images in `ghcr.io/vkolupaev/gpu-infra-lab`, target `lab`. An image cannot know its own
digest before push, so on the host export the full ref before `bench.py run`:

    export LAB_IMAGE=ghcr.io/vkolupaev/gpu-infra-lab:<tag>@<digest>

The digest is the one `docker push` prints (`<tag>: digest: sha256:...`), checked against
`docker buildx imagetools inspect <image>`. Date is the image build date.

| tag | digest | base | date | notes |
|---|---|---|---|---|
| `v0.32.0-cu129-fe18af2` | `sha256:55f609f2994f38fb7b9b04e388b26966051c4a39444af38b9c90268d44c10c8c` | `vllm/vllm-openai:v0.32.0-cu129@sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f` | 2026-10-21 | torchcodec cu13 bug, history not verified; `LAB_IMAGE` inside holds the base ref; commit fe18af2 predates a history rewrite |
| `v0.32.0-cu129-daffd21` | `sha256:8deda34450072ae5473ada23eb40a124ba15c4db7d2755a7ebc3ffe54637b85b` | `vllm/vllm-openai:v0.32.0-cu129@sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f` | 2026-10-25 | torchcodec removed; build fails if `vllm.multimodal` does not import |
