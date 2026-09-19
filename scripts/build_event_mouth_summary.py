"""Compatibility entry point for the report-driven event/mouth summarizer."""
from scripts.summarize_event_mouth_experiment import run, parser

if __name__ == '__main__':
    run(parser().parse_args())
