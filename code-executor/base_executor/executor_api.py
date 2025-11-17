from flask import Flask, request, jsonify
import subprocess, tempfile, os

app = Flask(__name__)

LANG = os.getenv("LANGUAGE", "python")

@app.route("/run", methods=["POST"])
def run_code():
    data = request.get_json()
    code = data.get("code", "")
    filename = data.get("filename", f"main.{LANG}")

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, "w") as f:
            f.write(code)

        try:
            cmd = []
            if LANG == "python":
                cmd = ["python3", filepath]
            elif LANG == "javascript":
                cmd = ["node", filepath]
            elif LANG == "typescript":
                cmd = ["npx", "ts-node", filepath]
            elif LANG == "java":
                subprocess.run(["javac", filepath], check=True)
                cmd = ["java", "-cp", tmpdir, filename.replace(".java", "")]
            elif LANG == "cpp":
                out = os.path.join(tmpdir, "a.out")
                subprocess.run(["g++", filepath, "-o", out], check=True)
                cmd = [out]
            elif LANG == "csharp":
                out = os.path.join(tmpdir, "program.exe")
                subprocess.run(["mcs", "-out:" + out, filepath], check=True)
                cmd = ["mono", out]
            elif LANG == "go":
                cmd = ["go", "run", filepath]
            elif LANG == "rust":
                cmd = ["rustc", filepath]
                cmd = ["./" + filepath.replace(".rs", "")]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return jsonify({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            })
        except subprocess.CalledProcessError as e:
            return jsonify({"error": str(e), "stderr": e.stderr})
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Execution timed out"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
