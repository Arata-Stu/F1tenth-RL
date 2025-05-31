## F1tenth rl repojitory
## 目次
- [プロジェクト概要](#プロジェクト概要)
- [前提条件](#前提条件)
- [環境セットアップ](#環境セットアップ)
- [実行方法](#実行方法)


## プロジェクト概要

F1tenth gym環境を用いた強化学習を行うプロジェクトです。

## 前提条件

- venv Python 3.11 
- Ubuntu 22.04

※ 上記の環境のみで検証しています。他の環境でも問題ないと考えられます。

## 環境セットアップ

### 1. 仮想環境の作成と有効化（Python）
```bash
## Python3.11 のインストール
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt update
# sudo apt install python3.11 python3.11-venv python3.11-dev

python3.11 -m venv env
source env/bin/activate  
```

### 2. 依存ライブラリのインストール
```bash
pip3 install -r requirements.txt
```