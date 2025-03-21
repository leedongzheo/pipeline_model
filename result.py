
from config import*
from train import*
def export():
    source_file1='last_model.pth'
    source_file2='best_model.pth'
    path=f"output_epoch{trainer.best_epoch}_dice{trainer.best_dice:.4f}"
    output_folder = os.path.join(BASE_OUTPUT,path)
    os.makedirs(output_folder, exist_ok=True)
    # Di chuyển
    exist_file_1=os.path.join(output_folder,source_file1)
    if os.path.exists(exist_file_1):
        os.remove(exist_file_1)
    shutil.move(source_file1, output_folder)
    if os.path.exists(source_file2):
        shutil.move(source_file2, output_folder)
        print(f"Đã di chuyển file tới: {output_folder}")

    # Kiểm tra xem source_file2 có tồn tại không trước khi di chuyển
    if os.path.exists(source_file2):
        shutil.move(source_file2, output_folder)
        print(f"Đã di chuyển file tới: {output_folder}")

    def tensor_to_float(value):
        if isinstance(value, torch.Tensor):
            return value.cpu().item()  # Chuyển tensor về CPU và lấy giá trị float
        elif isinstance(value, list):
            return [tensor_to_float(v) for v in value]  # Xử lý danh sách các tensor
        return value  # Nếu không phải tensor, giữ nguyên
    # Đường dẫn file checkpoint

    checkpoint_path = os.path.join(output_folder,source_file1)
    # source_file3 = "training_history.csv"  # File CSV hiện tại
    # csv_path_full = os.path.join(BASE_OUTPUT,source_file3)
    # Tải checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Đọc các giá trị từ checkpoint
    train_losses = tensor_to_float(checkpoint.get('train_losses', []))
    val_losses = tensor_to_float(checkpoint.get('val_losses', []))
    train_dices = tensor_to_float(checkpoint.get('train_dices', []))
    val_dices = tensor_to_float(checkpoint.get('val_dices', []))
    best_dice = tensor_to_float(checkpoint.get('best_dice', None))
    best_epoch = tensor_to_float(checkpoint.get('best_epoch', None))
    epoch = checkpoint.get('epoch', None)
    # start_epoch=checkpoint.get('start_epoch', None) + 1
    epochs = list(range(1, epoch + 1))
    # epochs = list(range(start_epoch, epoch + 1))

    new_data = pd.DataFrame({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_dices': train_dices,
        'val_dices': val_dices,
        'best_dice': [best_dice] * len(epochs),
        'best_epoch': [best_epoch] * len(epochs),
        'epoch': epochs
    })

    # Lưu vào file Excel
    output_path = 'training_history_current_1.csv'
    csv_path_currrent = os.path.join(output_folder,output_path)
    # csv_path_currrent = os.path.join(output_folder,output_path)
    new_data.to_csv(csv_path_currrent, index=False)

    print(f"[INFO] Training history saved to {csv_path_currrent}")
    df = pd.read_csv(csv_path_currrent, encoding='ISO-8859-1')  # Hoặc 'latin1', 'windows-1252'
    # df.info()

    # Plot Losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_losses'], label='Train Loss')
    plt.plot(df['epoch'], df['val_losses'], label='Valid Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Dice Coefficients
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_dices'], label='Train Dice')
    plt.plot(df['epoch'], df['val_dices'], label='Valid Dice')
    plt.title('Training and Validation Dice Coefficients')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    # Vẽ đồ thị
    plt.tight_layout()
        # Save the plot to a file
    source_file4="metrics_from_excel.png"
    output_metric = os.path.join(output_folder,source_file4)
    plt.savefig(output_metric, dpi=300)  # Tùy chỉnh độ phân giải với tham số dpi
    # source_folder = "/content/output" =>Bỏ

    # destination_folder = "/content/drive/MyDrive/ISIC/output_02-03-2025_PreTrain8With_Dice-CrossELoss_50loop"
    # destination_folder=os.path.join(destination_folder,path)
    # os.makedirs(destination_folder, exist_ok=True)
    # shutil.copytree(output_folder, destination_folder, dirs_exist_ok=True)

    plt.show()
    plt.close()
