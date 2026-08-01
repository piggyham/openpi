#!/usr/bin/env bash
set -euo pipefail

MISSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPENDENCY_DIR="${MISSION_DIR}/third_party/openarm_mujoco"
REPOSITORY="https://github.com/enactic/openarm_mujoco.git"
REVISION="8955afb54e4adfb59a236e2b4d15192b7a02865c"

if [[ -e "${DEPENDENCY_DIR}" && ! -d "${DEPENDENCY_DIR}/.git" ]]; then
  echo "Refusing to overwrite non-git path: ${DEPENDENCY_DIR}" >&2
  exit 1
fi

if [[ ! -d "${DEPENDENCY_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${DEPENDENCY_DIR}")"
  git clone --filter=blob:none "${REPOSITORY}" "${DEPENDENCY_DIR}"
fi

git -C "${DEPENDENCY_DIR}" fetch origin "${REVISION}"
git -C "${DEPENDENCY_DIR}" checkout --detach "${REVISION}"

MODEL_PATH="${DEPENDENCY_DIR}/v1/openarm_bimanual.xml"
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "OpenArm v1 model was not found at ${MODEL_PATH}" >&2
  exit 1
fi

ACTUAL_REVISION="$(git -C "${DEPENDENCY_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_REVISION}" != "${REVISION}" ]]; then
  echo "Revision mismatch: expected ${REVISION}, got ${ACTUAL_REVISION}" >&2
  exit 1
fi

echo "OpenArm v1 assets ready at ${DEPENDENCY_DIR}"
echo "Revision: ${ACTUAL_REVISION}"

