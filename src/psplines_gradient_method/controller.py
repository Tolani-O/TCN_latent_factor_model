import subprocess
import sys
import argparse


def checkout_git_commit(commit_hash):
    result = subprocess.run(['git', 'checkout', commit_hash], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error checking out commit {commit_hash}: {result.stderr.decode('utf-8').strip()}")
        sys.exit(1)
    return result.stdout.decode('utf-8').strip()


def run_python_script(command):
    if not command:
        print("No command provided.")
        sys.exit(1)

    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True) as proc:
        try:
            for line in proc.stdout:
                print(line.strip())  # Stream the output of the subprocess
            proc.wait()

            if proc.returncode != 0:
                print("Error running command!")
                print(proc.stderr.read().strip())
                sys.exit(1)
            else:
                print("Command completed successfully.")
                sys.exit(0)
        except Exception as e:
            print(f"Error occurred while running command: {e}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Checkout git commit and run a Python script.')
    parser.add_argument('commit_hash', help='The git commit hash to checkout.')
    parser.add_argument('command', help='The command to run after checking out the commit.')

    args = parser.parse_args()

    print(checkout_git_commit(args.commit_hash))

    # Run the command and stream its output
    run_python_script(args.command)
