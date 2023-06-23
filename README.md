# chatbot

Fine-tuning a chatbot on Messenger conversations

## Getting Started

### Framework

Set some environment variables:

```bash
export RUN_DIR=/path/to/run/dir  # This is where all your training logs and checkpoints will be written
export EVAL_RUN_DIR=/path/to/eval/run/dir  # This is where all your evaluation logs will be written
```

It's also a good idea to set these variables:

```bash
export DATA_DIR=/path/to/data/dir  # This is where your datasets are stored
export MODEL_DIR=/path/to/model/dir  # This is where your pretrained models are stored
```

Check out the documentation [here](https://ml.bolte.cc/getting_started.html).

### Dataset

1. Create a new directory called `$DATA_DIR/messenger`
2. Create a new JSON dump of your Messenger conversations from [here](https://www.messenger.com/dyi)
3. Download the ZIP file to the newly-created directory; it should have a path like `$DATA_DIR/messenger/facebook-{username}.zip`
4. Run `python -m chatbot.tasks.dataset` to preprocess the dataset

So ultimately you should have a directory structure like this:

```
$DATA_DIR
└── messenger
    ├── facebook-{username}.zip
    ├-- packed
    │   └── rwkv.bin
    └── messages
        └── inbox
            ├── {conversation_1}
            │   ├── message_1.json
            │   ├── message_2.json
            │   ├── ...
            │   └── message_n.json
            ├── {conversation_2}
            │   ├── message_1.json
            │   ├── message_2.json
            │   ├── ...
            │   └── message_n.json
            ├── ...
            └── {conversation_n}
                ├── message_1.json
                ├── message_2.json
                ├── ...
                └── message_n.json
```
