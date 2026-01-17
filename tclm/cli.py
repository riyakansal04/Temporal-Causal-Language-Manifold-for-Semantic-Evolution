import sys
from datetime import datetime, timedelta
from dateutil.parser import isoparse
from typing import Optional, Annotated

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .config import ensure_dirs, Paths
from .ingest.rss import fetch_rss_corpus
from .ingest.arxiv import fetch_arxiv_corpus
from .preprocess.clean import clean_corpus
from .preprocess.timebin import timebin_corpus
from .semantic.embed import embed_corpus
from .semantic.manifold import build_concept_trajectory
from .semantic.topics import extract_topics
from .causal.influence import estimate_causal_matrix, get_granger_summary_table
from .causal.consensus import fuse_consensus
from .forecast.models import forecast_trajectory
from .viz.plots import plot_trajectory, plot_causal_graph
from .training.train_embed import train_embedding, TrainConfig
from .evaluation.metrics import evaluate


app = typer.Typer(add_completion=False, rich_markup_mode="markdown")
console = Console()


def parse_date(s: Optional[str]) -> Optional[datetime]:
    return isoparse(s).replace(tzinfo=None) if s else None


@app.command()
def run(
    concept: str = typer.Option("privacy", help="Concept/term to track"),
    start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD"),
    freq: str = typer.Option("M", help="Time bin frequency (D/W/M/Q)"),
    max_docs: int = typer.Option(800, help="Max docs per source"),
    model_path: Optional[str] = typer.Option(None, help="Custom embedding model path"),
    forecasting_model: str = typer.Option("auto_arima", help="Forecasting model: auto_arima | prophet"),
):
    """Run end-to-end TCLM pipeline on public sources."""
    """Run end-to-end TCLM pipeline on public sources."""

    ensure_dirs()

    start_dt = parse_date(start)
    end_dt = parse_date(end)
    
    # If no dates provided, use a default range (last 6 months) to get more data
    if start_dt is None and end_dt is None:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=180)  # 6 months back
        console.print(f"[dim]No date range specified. Using default: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}[/dim]")
        console.print("[dim]Tip: Use --start and --end to specify a custom date range for more control.[/dim]")

    console.rule("Ingest")
    rss_df = fetch_rss_corpus(start_dt, end_dt, max_docs=max_docs)
    arxiv_df = fetch_arxiv_corpus(start_dt, end_dt, max_docs=max_docs)
    
    # Show date range of fetched data
    all_dfs = [df for df in [rss_df, arxiv_df] if not df.empty]
    if all_dfs:
        all_dates = pd.concat([df["published"] for df in all_dfs])
        min_date = all_dates.min()
        max_date = all_dates.max()
        date_span = (max_date - min_date).days
        console.print(f"[dim]Fetched data spans {date_span} days: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}[/dim]")
        if date_span < 30:
            console.print(f"[yellow]Warning: Data spans only {date_span} days. RSS feeds typically only return recent items (last 10-20).[/yellow]")
            console.print("[yellow]For more time bins, consider using historical datasets or specifying a longer date range if available.[/yellow]")

    table = Table(title="Ingested Docs")
    table.add_column("Source")
    table.add_column("Count", justify="right")
    table.add_row("rss", str(len(rss_df)))
    table.add_row("arxiv", str(len(arxiv_df)))
    console.print(table)
    
    # Check if we have any data
    total_docs = len(rss_df) + len(arxiv_df)
    if total_docs == 0:
        console.print("[red]Error: No documents were fetched.[/red]")
        if start_dt and end_dt:
            console.print(f"[yellow]Possible reasons:[/yellow]")
            console.print(f"  • RSS feeds typically only return the last 10-20 recent items")
            console.print(f"  • Requested date range ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}) may be too far in the past")
            console.print(f"  • arXiv API may not have data for the specified date range")
            console.print(f"\n[dim]Try:[/dim]")
            console.print(f"  • Remove --start and --end to use recent data (default: last 6 months)")
            console.print(f"  • Use a more recent date range (e.g., --start 2024-11-01)")
        else:
            console.print("[yellow]RSS feeds may be temporarily unavailable or have no recent items.[/yellow]")
        raise typer.Exit(code=1)

    console.rule("Preprocess")
    corpus = clean_corpus([rss_df, arxiv_df])
    
    if corpus.empty:
        console.print("[red]Error: All documents were filtered out during preprocessing.[/red]")
        raise typer.Exit(code=1)
    
    binned = timebin_corpus(corpus, freq=freq)

    console.rule("Embed")
    model_name = model_path or "sentence-transformers/all-MiniLM-L6-v2"
    embedded = embed_corpus(binned, model_name=model_name)

    console.rule("Topics & Trajectory")
    topics = extract_topics(embedded)
    traj = build_concept_trajectory(embedded, concept=concept)
    
    # Warn about insufficient data for trajectory analysis
    n_traj_points = len(traj)
    if n_traj_points < 10:
        console.print(f"[yellow]Warning: Only {n_traj_points} time points in trajectory.[/yellow]")
        console.print("[dim]Trajectory analysis requires 10+ time points for meaningful trends. With few points, differences appear as dramatic jumps rather than smooth trends.[/dim]")
        if n_traj_points < 5:
            console.print("[yellow]Forecast will use last value (insufficient data for ARIMA).[/yellow]")

    console.rule("Causal Influence & Consensus")
    
    # Check if we have enough time bins for causal analysis
    n_bins = embedded["time_bin"].nunique()
    if n_bins < 10:
        console.print(f"[yellow]Warning: Only {n_bins} time bins available. VAR/Granger causality analysis requires at least 10 time points for reliable results.[/yellow]")
        console.print("[dim]Consider using a longer time period or a finer time frequency (e.g., --freq W for weekly bins).[/dim]")
    
    causal_results = estimate_causal_matrix(embedded)
    consensus = fuse_consensus(traj, embedded)

    # Display Granger causality summary table
    console.rule("Granger Causality Summary")
    granger_table = get_granger_summary_table(causal_results)
    if len(granger_table) > 0:
        table = Table(title="Granger Causality Test Results")
        table.add_column("Cause", style="cyan")
        table.add_column("Effect", style="magenta")
        table.add_column("P-Value", justify="right", style="yellow")
        table.add_column("F-Statistic", justify="right", style="green")
        table.add_column("Significant", justify="center")
        table.add_column("Confidence (%)", justify="right", style="blue")
        
        # Show top 20 most significant relationships
        for _, row in granger_table.head(20).iterrows():
            sig_style = "[green]Yes[/green]" if row['Significant'] == 'Yes' else "[red]No[/red]"
            
            # Format p-value
            p_val = row['P-Value']
            if np.isnan(p_val) or np.isinf(p_val):
                p_str = "N/A"
            else:
                p_str = f"{p_val:.4f}"
            
            # Format F-statistic
            f_stat = row['F-Statistic']
            if np.isnan(f_stat) or np.isinf(f_stat) or f_stat < 0:
                f_str = "N/A"
            else:
                f_str = f"{f_stat:.2f}"
            
            # Format confidence
            conf = row['Confidence (%)']
            if np.isnan(conf) or np.isinf(conf):
                conf_str = "N/A"
            else:
                conf_str = f"{conf:.1f}"
            
            table.add_row(
                row['Cause'],
                row['Effect'],
                p_str,
                f_str,
                sig_style,
                conf_str
            )
        console.print(table)
        if len(granger_table) > 20:
            console.print(f"[dim]... and {len(granger_table) - 20} more relationships[/dim]")
    else:
        console.print("[yellow]No causal relationships to display[/yellow]")

    # Generate and save causal graph visualization
    console.rule("Causal Graph Visualization")
    graph_path = plot_causal_graph(causal_results)
    console.print(f"Saved causal graph: {graph_path}")

    console.rule("Forecast")
    fcst = forecast_trajectory(traj, steps=6, model_type=forecasting_model)

    console.rule("Plot")
    plot_path = plot_trajectory(traj, fcst, concept)
    console.print(f"Saved plot: {plot_path}")

    console.rule("Summary")
    console.print({
        "docs": {"rss": len(rss_df), "arxiv": len(arxiv_df)},
        "bins": embedded["time_bin"].nunique(),
        "topics": len(topics),
        "causal_sources": len(causal_results.sources),
        "significant_relationships": len(granger_table[granger_table['Significant'] == 'Yes']) if len(granger_table) > 0 else 0,
        "forecast_steps": len(fcst),
        "forecast_model": forecasting_model,
    })


@app.command()
def train(
    epochs: int = typer.Option(1, help="Training epochs"),
    lr: float = typer.Option(2e-5, help="Learning rate"),
    output_dir: Optional[str] = typer.Option(None, help="Where to save the model"),
    start: Optional[str] = typer.Option(None, help="Start date"),
    end: Optional[str] = typer.Option(None, help="End date"),
    max_docs: int = typer.Option(2000, help="Max docs per source"),
):
    """Fine-tune the embedding model with weak supervision (title<->text pairs)."""
    ensure_dirs()
    start_dt = parse_date(start)
    end_dt = parse_date(end)
    console.rule("Ingest for Training")
    rss_df = fetch_rss_corpus(start_dt, end_dt, max_docs=max_docs)
    arxiv_df = fetch_arxiv_corpus(start_dt, end_dt, max_docs=max_docs)
    corpus = clean_corpus([rss_df, arxiv_df])
    cfg = TrainConfig(epochs=epochs, lr=lr)
    if output_dir:
        cfg.output_dir = Paths.root / output_dir
    out = train_embedding(corpus, cfg)
    console.print(f"Saved fine-tuned model to: {out}")


@app.command()
def eval(
    concept: str = typer.Option("privacy", help="Concept/term to track"),
    start: Optional[str] = typer.Option(None, help="Start date"),
    end: Optional[str] = typer.Option(None, help="End date"),
    freq: str = typer.Option("M", help="Time bin frequency"),
    max_docs: int = typer.Option(800, help="Max docs per source"),
    model_path: Optional[str] = typer.Option(None, help="Custom embedding model path"),
):
    """Run evaluation metrics: retrieval recall@1, silhouette, forecast backtest."""
    ensure_dirs()
    start_dt = parse_date(start)
    end_dt = parse_date(end)
    rss_df = fetch_rss_corpus(start_dt, end_dt, max_docs=max_docs)
    arxiv_df = fetch_arxiv_corpus(start_dt, end_dt, max_docs=max_docs)
    corpus = clean_corpus([rss_df, arxiv_df])
    binned = timebin_corpus(corpus, freq=freq)
    model_name = model_path or "sentence-transformers/all-MiniLM-L6-v2"
    embedded = embed_corpus(binned, model_name=model_name)
    traj = build_concept_trajectory(embedded, concept=concept)
    res = evaluate(embedded, traj)
    table = Table(title="Evaluation Metrics")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("retrieval_recall@1", f"{res.retrieval_recall_at1:.3f}")
    table.add_row("clustering_silhouette", f"{res.clustering_silhouette:.3f}")
    table.add_row("backtest_rmse", f"{res.backtest_rmse:.3f}")
    table.add_row("backtest_mae", f"{res.backtest_mae:.3f}")
    table.add_row("backtest_mape", f"{res.backtest_mape:.3f}")
    console.print(table)


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}\n{type(e).__name__}")
        sys.exit(1)


