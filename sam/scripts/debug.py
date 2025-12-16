import sys
import os
import torch
from timm import create_model

# Setup path import
sys.path.append(os.getcwd())
try:
    import repvit_sam.modeling
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    import repvit_sam.modeling

def main():
    print("🔍 Menganalisis Struktur RepViT...")
    model = create_model('repvit')
    model.eval()

    # Ambil satu block contoh (biasanya ada di dalam features atau token_mixer)
    print("\n--- STRUKTUR BLOCK PERTAMA ---")
    found = False
    for name, m in model.named_modules():
        # Cari modul yang punya method switch_to_deploy
        if hasattr(m, 'switch_to_deploy') or hasattr(m, 'fuse'):
            print(f"Modul ditemukan: {name}")
            print(f"Tipe Class: {type(m)}")
            print("Atribut sebelum fuse:")
            print(f" - deploy: {getattr(m, 'deploy', 'N/A')}")
            print(f" - rbr_dense: {'ADA' if hasattr(m, 'rbr_dense') else 'TIDAK'}")
            print(f" - rbr_reparam: {'ADA' if hasattr(m, 'rbr_reparam') else 'TIDAK'}")
            
            # Coba jalankan fuse manual pada satu block ini
            if hasattr(m, 'switch_to_deploy'):
                print("-> Menjalankan switch_to_deploy()...")
                m.switch_to_deploy()
            elif hasattr(m, 'fuse'):
                print("-> Menjalankan fuse()...")
                m.fuse()
            
            print("Atribut SESUDAH fuse:")
            print(f" - deploy: {getattr(m, 'deploy', 'N/A')}")
            print(f" - rbr_reparam: {'ADA' if hasattr(m, 'rbr_reparam') else 'TIDAK'}")
            
            # Cek apakah rbr_dense hilang?
            print(f" - rbr_dense (Harusnya hilang): {'MASIH ADA' if hasattr(m, 'rbr_dense') else 'SUDAH HILANG'}")
            
            found = True
            break
    
    if not found:
        print("❌ TIDAK DITEMUKAN block yang bisa di-fuse!")
        print("Coba print seluruh model:")
        print(model)

if __name__ == "__main__":
    main()