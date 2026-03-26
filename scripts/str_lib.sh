#!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

message() {
  local level="$1"
  shift
  case "$level" in
    info)    echo -e "${CYAN}[info]${NC} $*" ;;
    success) echo -e "${GREEN}[ok]${NC} $*" ;;
    warning) echo -e "${YELLOW}[warn]${NC} $*" ;;
    error)   echo -e "${RED}[error]${NC} $*" ;;
    *)       echo "$level $*" ;;
  esac
}
