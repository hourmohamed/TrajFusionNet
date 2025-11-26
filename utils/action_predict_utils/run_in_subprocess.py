import subprocess

def run_and_capture_model_path(command: list):
    captured_output, model_path = [], ""
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in iter(process.stdout.readline, ''):
            print(line, end="")
            captured_output.append(line)
    prefix = "Model saved under: "
    for l in reversed(captured_output):
        print("loop 2")
        if prefix in l:
            model_path = l[len(prefix):].strip()
    return model_path
