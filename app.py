"""Entry point so `flask run` can auto-discover the application."""

from src.dsa_median.webapp import app

__all__ = ["app"]

if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)
