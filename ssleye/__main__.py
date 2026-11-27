import argparse

from .server import serve


def main():
    ap = argparse.ArgumentParser(prog="python -m ssleye")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ui = sub.add_parser("ui", help="open the dashboard server")
    ui.add_argument("--dir", default="ssleye_runs")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=5005)
    a = ap.parse_args()
    if a.cmd == "ui":
        serve(a.dir, a.host, a.port)


if __name__ == "__main__":
    main()
