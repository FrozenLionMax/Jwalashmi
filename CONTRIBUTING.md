# Contributing to JWALASHMI

Thank you for your interest in contributing to JWALASHMI! This project aims to build a reliable solar flare early warning system using data from ISRO's Aditya-L1 mission.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/Jwalashmi.git
   cd Jwalashmi
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the dashboard:
   ```bash
   python server.py
   ```

## Project Structure

| Directory | Description |
|-----------|-------------|
| `dashboard/` | Frontend mission control UI (single-file SPA) |
| `src/data/` | FITS data loading and extraction |
| `src/features/` | Physics-informed feature engineering |
| `src/model/` | ML model architecture, training, and evaluation |
| `src/nowcasting/` | Real-time flare detection engine |

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or suggest features
- Include steps to reproduce for bugs
- Include expected vs actual behavior

### Code Contributions
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Test locally with `python server.py`
4. Commit with clear messages: `git commit -m "Add: description of change"`
5. Push and open a Pull Request

### Areas We Need Help With
- **GOES Pre-training**: Pipeline to use 50 years of GOES XRS data for transfer learning
- **Production Deployment**: Gunicorn/Nginx configuration for production serving
- **WebSocket Streaming**: Replace polling with real-time WebSocket for live flux data
- **Reliability Diagrams**: Calibration curve visualization for model validation
- **Additional Instruments**: Integration with other Aditya-L1 payloads (SUIT, VELC)

## Code Style
- Python: Follow PEP 8
- JavaScript: Use consistent ES6+ syntax
- Keep functions focused and documented
- Use meaningful variable names

## Data
- FITS data files are NOT included in the repository (too large)
- Place SoLEXS files in `Solexs/` and HEL1OS files in `Helios/`
- The dashboard works in simulation mode without real data

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
