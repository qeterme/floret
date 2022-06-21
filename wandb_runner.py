import typer
import os
from subprocess import PIPE, Popen

import wandb

app = typer.Typer()


floret_cmd = "{path}/floret cbow -dim {dim} -mode {mode} -bucket {bucket} -minn {minn} -maxn {maxn} -minCount {mincount} -neg {neg} -hashCount {hashcount} -lr {lr} -thread {thread} -epoch {epoch} -input {input} -output {output}"
proc = None

@app.command()
def main(floret_path: str, dim: int, mode: str, bucket: int, minn: int, maxn: int, mincount: int, neg: int, hashcount: int, lr: float, thread: int, epoch: int, input: str, output: str, wandb_project: str = "floret", wandb_dry: bool = False):
    if wandb_dry:
        os.environ['WANDB_MODE'] = 'dryrun'

    run_name = f"floret bucket:{bucket} minn:{minn} maxn:{maxn} mincount:{mincount} neg:{neg} lr:{lr} epoch:{epoch}"

    floret_train_cmd = floret_cmd.format(path=floret_path, dim=dim, mode=mode, bucket=bucket, minn=minn, maxn=maxn,
                                   mincount=mincount, neg=neg, hashcount=hashcount, lr=lr, thread=thread, epoch=epoch, input=input, output=output)
    print(floret_train_cmd)

    wandb.init(name=run_name, project=wandb_project)
    config = {
        "dimension": dim,
        "bucket": bucket,
        "minn": minn,
        "maxn": maxn,
        "mincount": mincount,
        "neg": neg,
        "hashcount": hashcount,
        "learning_rate": lr,
        "thread": thread,
        "epochs": epoch,
        "input": input,
        "output": output
    }

    wandb.config.update(config)

    proc = Popen(floret_train_cmd, shell=True, stderr=PIPE, universal_newlines=True)
    while proc.poll() is None:
        for line in proc.stderr: 
            if ("\t" in line):
                parts_of_line = line.split("\t")
                print(f"\rProgress: {float(parts_of_line[0].rstrip()):5.1f}% words/sec/thread: {int(parts_of_line[1].rstrip()):7d} lr: {float(parts_of_line[2].rstrip()):9.6f} loss: {float(parts_of_line[3].rstrip()):9.6f} ETA: {parts_of_line[4].rstrip()}", flush=True, end='')
                log_to_wandb(float(parts_of_line[2].rstrip()), float(parts_of_line[3].rstrip()))
            else:
                print(line, end='')


def sigterm_handler(signal, frame):
    if proc is not None:
        proc.kill()
    exit(0)


def log_to_wandb(learning_rate: float, loss: float):
    wandb.log({
        "loss": loss,
        "learning_rate": learning_rate,
        })


if __name__ == "__main__":
    app()
