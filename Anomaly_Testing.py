#In This file we will check every Tensor list to see if they are Anomalys
#This method may cause confusion and is hard to determine the thereshold value due to MinMaxScaler in our training code which limits Tensor value from -1 to +1

def evaluate_and_identify_anomalies(eval_model, data_source, threshold=0.2, output_csv_filename='csv_anomaly.csv'):
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = 1000
    output_data = []  
    anomalies = []  

    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)

            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()

            prediction = output[-output_window:].cpu().numpy()
            actual = targets[-output_window:].cpu().numpy()
            diff = np.abs(prediction - actual)

            anomaly_indicator = (diff > threshold).astype(int)

            output_data.append({'Prediction': prediction, 'Actual': actual})
            anomalies.extend(anomaly_indicator.tolist())
