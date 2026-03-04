import torch
import os


def inspect_brain(file_path):
    print(f"\n🧠 Abrindo o Córtex: {file_path}")
    print("=" * 80)

    if not os.path.exists(file_path):
        print(f"❌ Arquivo não encontrado: {file_path}")
        return

    try:
        # weights_only=True garante que estamos apenas lendo tensores, por segurança
        state_dict = torch.load(file_path, weights_only=True, map_location='cpu')

        print(f"📊 Total de Matrizes/Camadas: {len(state_dict)}\n")
        print(f"{'NOME DA CAMADA NEURAL':<40} | {'FORMATO (SHAPE)':<20} | {'PESO MÉDIO':<10}")
        print("-" * 80)

        total_parameters = 0

        for layer_name, weight_tensor in state_dict.items():
            # Conta os parâmetros
            params = weight_tensor.numel()
            total_parameters += params

            # Formata a exibição
            shape_str = str(list(weight_tensor.shape))
            mean_val = weight_tensor.mean().item()

            print(f"{layer_name:<40} | {shape_str:<20} | {mean_val:>8.4f}")

        print("-" * 80)
        print(f"🧠 Total de Parâmetros Treináveis: {total_parameters:,}")

    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")


if __name__ == "__main__":
    # Vamos ler os 3 modelos que compõem o sistema motor da Aura
    brain_folder = "./aura_brain"

    inspect_brain(f"{brain_folder}/world_model.pth")
    inspect_brain(f"{brain_folder}/policy.pth")
    inspect_brain(f"{brain_folder}/inverse.pth")