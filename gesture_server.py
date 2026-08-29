from flask import Flask, request, jsonify

app = Flask(__name__)


@app.post("/control")
def control():
    data = request.get_json()
    command = data.get("command")

    print("received:", command)

    return jsonify({"ok": True})


app.run(host="127.0.0.1", port=5001)