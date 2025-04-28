from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func
import os

app = Flask(__name__)

# Configure CORS for your Cloudflare frontend domain(s)
CLOUDFLARE_FRONTEND_DOMAINS = ["https://qual-bench-ai-web.jacobautus2.workers.dev/"]  # Replace with your actual domain(s)
CORS(app, resources={r"/*": {"origins": CLOUDFLARE_FRONTEND_DOMAINS + ["http://localhost:3000"]}})

# SQLAlchemy setup
DATABASE_URL = os.environ.get('mysql+pymysql://capstoneteam:csusmcapstone@34.102.56.241:3306/qualbenchaidb')
if DATABASE_URL:
    engine = create_engine(DATABASE_URL, echo=False)  # Set echo to False in production
else:
    # Fallback for local development - REMOVE OR UPDATE FOR PRODUCTION
    engine = create_engine('mysql+pymysql://root:csusmcapstone@34.102.56.241:3306/qualbenchaidb', echo=True)

Session = sessionmaker(bind=engine)
Base = declarative_base()

# SQLAlchemy Models
class Model(Base):
    __tablename__ = 'models'
    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(255), nullable=False)
    website_url = Column(String(255))
    article_url = Column(String(255))
    benchmarks = relationship("Benchmark", back_populates="model")


class Device(Base):
    __tablename__ = 'devices'
    device_id = Column(Integer, primary_key=True, autoincrement=True)
    device_name = Column(String(255), nullable=False)
    chipset = Column(String(255), nullable=False)
    operating_system = Column(String(255), nullable=False)
    benchmarks = relationship("Benchmark", back_populates="device")


class Benchmark(Base):
    __tablename__ = 'benchmarks'
    benchmark_id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('models.model_id'), nullable=False)
    device_id = Column(Integer, ForeignKey('devices.device_id'), nullable=False)
    runtime = Column(String(255), nullable=False)
    mAP = Column(Float)
    AP_50 = Column(Float)
    AP_75 = Column(Float)
    AP_small = Column(Float)
    AP_medium = Column(Float)
    AP_large = Column(Float)
    AR = Column(Float)
    AR_50 = Column(Float)
    AR_75 = Column(Float)
    AR_small = Column(Float)
    AR_medium = Column(Float)
    AR_large = Column(Float)
    inference_time = Column(Float)  # Changed to Float
    memory_usage = Column(Float)
    min_memory = Column(Float)
    compute_units_cpu = Column(Integer)
    compute_units_gpu = Column(Integer)
    compute_units_npu = Column(Integer)
    model = relationship("Model", back_populates="benchmarks")
    device = relationship("Device", back_populates="benchmarks")


# API Routes
@app.route('/api/devices', methods=['GET'])
def get_devices():
    try:
        with Session() as session:
            devices = session.query(Device).all()
            return jsonify([
                {
                    "device_id": d.device_id,
                    "device_name": d.device_name,
                    "chipset": d.chipset,
                    "operating_system": d.operating_system
                } for d in devices
            ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/modeldata', methods=['GET'])
def get_modeldata():
    device_id = request.args.get('device_id', type=int)
    if device_id is None:
        return jsonify({"error": "Missing device_id parameter"}), 400
    try:
        with Session() as session:
            results = (
                session.query(Benchmark, Model)
                .join(Model, Benchmark.model_id == Model.model_id)
                .filter(Benchmark.device_id == device_id)
                .all()
            )
            model_data = {}
            for benchmark, model in results:
                model_data[model.model_name] = {
                    "mAP": benchmark.mAP,
                    "inference_time": benchmark.inference_time,
                    "memory_usage": benchmark.memory_usage,
                    "website_url": model.website_url,
                    "article_url": model.article_url,
                    "AP_small": benchmark.AP_small,
                    "AP_medium": benchmark.AP_medium,
                    "AP_large": benchmark.AP_large,
                    "AP_50": benchmark.AP_50,
                    "AP_75": benchmark.AP_75,
                    "AR": benchmark.AR,
                    "AR_50": benchmark.AR_50,
                    "AR_75": benchmark.AR_75,
                    "AR_small": benchmark.AR_small,
                    "AR_medium": benchmark.AR_medium,
                    "AR_large": benchmark.AR_large,
                    "runtime": benchmark.runtime,
                    "memory_usage": benchmark.memory_usage,
                    "min_memory": benchmark.min_memory,
                    "compute_units_npu": benchmark.compute_units_npu,
                    "compute_units_cpu": benchmark.compute_units_cpu,
                    "compute_units_gpu": benchmark.compute_units_gpu,
                }
            return jsonify({"data": model_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    device_id = request.args.get('device_id', type=int)
    metric = request.args.get('metric', type=str)
    sort_order = request.args.get('sort_order', default='default', type=str)

    if device_id is None:
        return jsonify({"error": "Missing device_id parameter"}), 400

    allowed_metrics = [
        'runtime',
        'inference_time',
        'mAP', 'min_memory', 'memory_usage',
        'compute_units_cpu', 'compute_units_gpu', 'compute_units_npu',
        'AP_small', 'AP_medium', 'AP_large',
        'AP_50', 'AP_75', 'AR', 'AR_50', 'AR_75',
        'AR_small', 'AR_medium', 'AR_large'
    ]

    if metric not in allowed_metrics:
        return jsonify({"error": "Invalid metric"}), 400

    allowed_sort_orders = ['default', 'asc', 'desc']
    if sort_order not in allowed_sort_orders:
        return jsonify({"error": "Invalid sort_order"}), 400

    try:
        with Session() as session:
            subquery = (
                session.query(
                    Benchmark.model_id,
                    func.min(Benchmark.benchmark_id).label('min_benchmark_id')
                )
                .filter(Benchmark.device_id == device_id)
                .group_by(Benchmark.model_id)
                .subquery()
            )

            query = (
                session.query(Benchmark, Model)
                .join(subquery, Benchmark.benchmark_id == subquery.c.min_benchmark_id)
                .join(Model, Benchmark.model_id == Model.model_id)
            )

            if sort_order == 'asc':
                query = query.order_by(getattr(Benchmark, metric).asc())
            elif sort_order == 'desc':
                query = query.order_by(getattr(Benchmark, metric).desc())

            results = query.all()
            labels = [model.model_name for _, model in results]
            values = [getattr(benchmark, metric) for benchmark, _ in results]

            return jsonify({"labels": labels, "values": values})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
