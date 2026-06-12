# doudizhu-tornado

Play **Dou Di Zhu** (斗地主, "Fight the Landlord") in your browser against the deep reinforcement learning agents from [doudizhu-C](https://github.com/qq456cvb/doudizhu-C) — the Combinatorial Q-Learning (CQL) agents published at [AIIDE 2020](https://ojs.aaai.org/index.php/AIIDE/article/view/7445).

This is an adaptation of [mailgyc/doudizhu](https://github.com/mailgyc/doudizhu) (an HTML5 + Tornado + MySQL Dou Di Zhu game): the original scripted bots are replaced with trained CQL agents, so the two robot seats at the table are driven by the neural network.

![](screenshot/a.png)
![](screenshot/c.png)
![](screenshot/d.png)

## How It Works

- **Frontend** (`static/`, `templates/poker.html`) — a [Phaser](https://phaser.io/) HTML5 client that talks to the server over a WebSocket (`/ws`).
- **Server** (`app.py`, `handlers/`, `core/`) — a Tornado application managing rooms and tables (`core/room.py`, `core/table.py`), game rules (`core/rule.py`), and user accounts backed by MySQL (`schema.sql`).
- **AI players** (`core/robot.py`, `core/predictor.py`, `core/DQNModel.py`) — each table loads Tensorpack `OfflinePredictor`s with the CQL Dueling-DQN network. At every turn the agent enumerates card-group decompositions of its hand (via the compiled `env` module from doudizhu-C), encodes them with pretrained auto-encoder embeddings (`core/res/encoding.npy`), and picks the best move with its two-stage Q-network. AI players join through a loopback socket (`handlers/loopback.py`) and behave exactly like remote clients.
- An endgame Monte-Carlo tree search (`mct` from [doudizhu-baseline](https://github.com/qq456cvb/doudizhu-baseline)) is wired into `core/robot.py` but disabled by default (`MCT_THRESH = 0`).

## Setup

Tested with Python 3.6, TensorFlow 1.x and Tensorpack (same environment as [doudizhu-C](https://github.com/qq456cvb/doudizhu-C#installation)).

### 1. Python dependencies

```bash
pip3 install -r requirements.txt
pip3 install tensorflow==1.13.1 tensorpack==0.8.5
```

### 2. MySQL

Create the database and tables:

```bash
mysql --user=root -p < schema.sql
```

### 3. Compiled game engine

The AI imports the C++ `env` module built from doudizhu-C (and optionally `mct` from doudizhu-baseline for the MCTS endgame solver). Build doudizhu-C following [its instructions](https://github.com/qq456cvb/doudizhu-C#installation) and copy the resulting Python extension(s) into a `build/` folder at this repository's root:

```
doudizhu-tornado/
└── build/
    ├── env.cpython-36m-x86_64-linux-gnu.so   # from doudizhu-C
    └── mct.cpython-36m-x86_64-linux-gnu.so   # optional, from doudizhu-baseline
```

### 4. Pretrained model

Download the CQL checkpoint from [Hugging Face](https://huggingface.co/qq456cvb/doudizhu-C) into `core/res/` (the server loads `core/res/model-302500`):

```bash
hf download qq456cvb/doudizhu-C model-302500.index model-302500.data-00000-of-00001 --local-dir core/res
```

A [Google Drive mirror](https://drive.google.com/drive/folders/1YTNR5JYNgNfQpQ9DwhQ3ClcyeXKn_17b?usp=sharing) is also available. The auto-encoder card embeddings (`core/res/encoding.npy`) already ship with this repository.

## Run

```bash
python3 app.py --password=your_database_password
```

Then open `http://localhost:8080` in a browser (use `--port` to change the port; see `settings/base.py` for all options). Register an account, sit at a table, and two AI players will join automatically.

## Related Projects

- [doudizhu-C](https://github.com/qq456cvb/doudizhu-C) — training code and pretrained models for the CQL agents (AIIDE 2020).
- [doudizhu-baseline](https://github.com/qq456cvb/doudizhu-baseline) — Monte-Carlo-Tree-Search baseline used for the optional endgame solver.
- [mailgyc/doudizhu](https://github.com/mailgyc/doudizhu) — the original web game this server is adapted from.

## Citation

If you use the agents in your work, please consider citing:

```bibtex
@inproceedings{you2020combinatorial,
  title={Combinatorial Q-Learning for Dou Di Zhu},
  author={You, Yang and Li, Liangwei and Guo, Baisong and Wang, Weiming and Lu, Cewu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
  volume={16},
  number={1},
  pages={301--307},
  year={2020}
}
```
