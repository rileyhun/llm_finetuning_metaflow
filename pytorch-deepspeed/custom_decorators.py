from functools import wraps

dir = "data"

def magicdir(f):
    artifact = "magicdir"

    @wraps(f)
    def func(self):
        from io import BytesIO
        from tarfile import TarFile

        existing = getattr(self, artifact, None)
        if existing:
            buf = BytesIO(existing)
            with TarFile(mode="r", fileobj=buf) as tar:
                tar.extractall()
        f(self)
        buf = BytesIO()
        with TarFile(mode="w", fileobj=buf) as tar:
            tar.add(dir)
        setattr(self, artifact, buf.getvalue())

    return func


def pip(libraries):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            import os
            import subprocess
            import sys

            IS_AWS_BATCH = os.getenv("AWS_BATCH_JOB_ID", None)

            if IS_AWS_BATCH:
                for library, version in libraries.items():
                    print("Pip Install:", library, version)
                    if version != "":
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "pip",
                                "install",
                                library + "==" + version,
                            ]
                        )
                    else:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", library]
                        )

            return function(*args, **kwargs)

        return wrapper

    return decorator


def enable_decorator(dec, flag):
    def decorator(func):
        if flag:
            return dec(func)
        return func

    return decorator
