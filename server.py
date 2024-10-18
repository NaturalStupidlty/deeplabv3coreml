from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/post', methods=['POST'])
def post_data():
    data = request.get_json()
    print(f"Received data: {data}")
    return jsonify({"status": "success", "data_received": data})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
