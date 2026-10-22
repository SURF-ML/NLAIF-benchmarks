ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Mirror vllm-bench.def: vllm bench deps + Ray for multi-node (TP×PP) serving.
RUN pip install --root-user-action=ignore vllm[bench] "ray[cgraph]>=2.43.0"

# CUDA-only CuPy alignment — match torch's CUDA major so Ray's compiled-graph comms get a single
# consistent CuPy. On the ROCm base image torch.version.cuda is None, so this is skipped; the
# platform is detected from torch (not a build arg) so one containerfile builds for both vendors.
RUN CUDA_MAJOR=$(python3 -c "import torch,re; v=torch.version.cuda; print(re.match(r'(\d+)', v).group(1) if v else '')") && \
    if [ -n "$CUDA_MAJOR" ]; then \
        pip uninstall --root-user-action=ignore -y cupy-cuda12x cupy-cuda13x || true; \
        pip install --root-user-action=ignore --no-cache-dir "cupy-cuda${CUDA_MAJOR}x"; \
    else \
        echo "Non-CUDA (ROCm) base image detected — skipping CuPy install."; \
    fi

RUN printf '%s\n' \
    "alias ls='ls --color=auto'" \
    "alias ll='ls -la'" \
    "alias l.='ls -d .* --color=auto'" \
    >> /etc/bashrc

ENV LC_ALL=C
ENV LD_PRELOAD=

ENTRYPOINT []
CMD []
