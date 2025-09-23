import argparse
import os
import sys
import random
from pathlib import Path
import cv2


def compute_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count <= 0:
        # Fallback: contar usando grab (rápido, sem decodificar)
        n = 0
        while True:
            grabbed = cap.grab()
            if not grabbed:
                break
            n += 1
        count = n

    cap.release()
    return count


def extract_random_frames(video_path: str, num_frames: int, out_dir: Path, seed: int | None = None, prefix: str = "frame") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = compute_frame_count(video_path)
    if total_frames <= 0:
        raise RuntimeError("Não foi possível determinar o número de frames do vídeo.")

    if num_frames <= 0:
        raise ValueError("O número de frames deve ser maior que zero.")

    if num_frames > total_frames:
        raise ValueError(f"Solicitado {num_frames} frames, mas o vídeo possui apenas {total_frames}.")

    rng = random.Random(seed)
    pending = set(rng.sample(range(total_frames), num_frames))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    width_digits = max(6, len(str(total_frames)))
    saved_paths: list[Path] = []

    def save_frame(idx: int, frame) -> Path:
        name = f"{prefix}_{idx:0{width_digits}d}.png"
        path = out_dir / name
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Falha ao salvar a imagem: {path}")
        return path

    # Tentar extrair os índices sorteados; se algum falhar, tentar índices alternativos
    failed: list[int] = []
    for idx in sorted(pending):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            saved_paths.append(save_frame(idx, frame))
        else:
            failed.append(idx)

    # Recuperação: tentar novos índices para substituir falhas
    if failed:
        remaining = set(range(total_frames)) - pending
        tries = 0
        max_tries = 4 * len(failed)
        while failed and remaining and tries < max_tries:
            tries += 1
            idx = rng.choice(tuple(remaining))
            remaining.remove(idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                saved_paths.append(save_frame(idx, frame))
                failed.pop()  # substitui uma falha

        if failed:
            cap.release()
            raise RuntimeError(f"Não foi possível extrair {len(failed)} frame(s) após tentativas de recuperação.")

    cap.release()
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Extrai N frames aleatórios de um vídeo (.avi).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="Caminho para o arquivo de vídeo (.avi)")
    parser.add_argument("num", type=int, help="Número de frames aleatórios a extrair (N)")
    parser.add_argument("-o", "--out", default=None, help="Diretório de saída")
    parser.add_argument("--seed", type=int, default=None, help="Semente do gerador de números aleatórios")
    parser.add_argument("--prefix", default="frame", help="Prefixo para os arquivos de imagem gerados")

    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"Erro: arquivo não encontrado: {video_path}")
        sys.exit(1)

    # Aviso leve sobre extensão, mas não bloqueia outros formatos
    if not video_path.lower().endswith(".avi"):
        print("Aviso: o caminho não termina com .avi; tentando abrir mesmo assim.")

    out_dir = Path(args.out) if args.out else Path(f"{Path(video_path).stem}_frames")
    try:
        saved = extract_random_frames(video_path, args.num, out_dir, seed=args.seed, prefix=args.prefix)
    except Exception as e:
        print(f"Erro: {e}")
        sys.exit(1)

    print(f"Frames salvos em: {out_dir.resolve()}")
    print(f"Total: {len(saved)}")


if __name__ == "__main__":
    main()