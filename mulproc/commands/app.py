import typer
from pathlib import Path
app = typer.Typer()

@app.command()
def celeba():
    typer.echo("celeba command")
    
train_index = 190
retain_index = 1250
unseen_index = 4855

@app.command(name="split")
def split(
    raw_h5: Path = typer.Argument(..., help="Path to the raw h5 file"),
    train_index: int = typer.Option(190,"--train","-s", help="Index of the train split"),
    retain_index: int = typer.Option(1250,"--retain","-r", help="Index of the retain split"),
    unseen_index: int = typer.Option(4855,"--unseen","-u", help="Index of the unseen split")
):
    typer.echo("split command")