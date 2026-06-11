# doudizhu-tornado

<!-- README refined by Cursor -->

A Tornado-based Dou Di Zhu mini-server adapted for interactive play.

## Overview

This repository contains Python, JavaScript, HTML code from an older research, course, or prototype project. The README has been refreshed to make the repository easier to scan while preserving the original notes below.

## Repository Contents

- `core/`
- `db/`
- `handlers/`
- `screenshot/`
- `settings/`
- `static/`
- `templates/`
- `tests/`

## Setup

- Install Python dependencies with `pip install -r requirements.txt`.

## Usage

- inspect the top-level Python entry points: `__init__.py`, `app.py`, `captcha.py`, `config.py`, `debug.py`

## Data and Artifacts

No new large artifact is stored in this repository. If a dataset or checkpoint is required, follow the links and notes in the original section below.

## Status

This is a `Batch B` cleanup pass for a legacy repository. Commands may require dependency/version adjustments on a modern machine.

## License

No explicit license file was found in this checkout; check the original project context before reusing code.

## Original Notes

# 斗地主
This repo is adapted from https://github.com/mailgyc/doudizhu

斗地主游戏, html5, python, mysql 实现

[![Build Status](https://travis-ci.org/mailgyc/doudizhu.svg?branch=master)](https://travis-ci.org/mailgyc)
[![Coverage Status](https://coveralls.io/repos/github/mailgyc/doudizhu/badge.svg?branch=master)](https://coveralls.io/github/mailgyc/doudizhu?branch=master)

运行步骤:

1.安装MySQL数据库, 执行:

    mysql --user=root -p < schema.sql

2.安装Python3.6, 执行:

    pip3 install -r requirements.txt

    python3 app.py --password=your_database_password

浏览器访问即可,端口默认8080

运行截图

![](https://raw.githubusercontent.com/mailgyc/doudizhu/master/screenshot/a.png)
![](https://raw.githubusercontent.com/mailgyc/doudizhu/master/screenshot/c.png)
![](https://raw.githubusercontent.com/mailgyc/doudizhu/master/screenshot/d.png)
