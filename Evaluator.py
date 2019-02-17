"""
Created on 2018/10/21 by Chunhui Yin(yinchunhui.ahu@gmail.com).
Description:Evaluating experimental results.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(model, x_test, y_test):
    predictions = model.predict(x_test, batch_size=256, verbose=0)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return mae, rmse


# Save the evaluation results into file
def saveResult(resultPath, dataType, density, result, metrics):
    if density:
        fileID = open('%s/%s_result_%.2f.txt' % (resultPath, dataType, density), 'w')
    else:
        fileID = open('%s/%s_result.txt' % (resultPath, dataType,), 'w')
    fileID.write('Metric: ')
    for metric in metrics:
        fileID.write('| %s\t' % metric)

    avgResult = np.average(result, axis=0)
    fileID.write('\nAvg:\t')
    np.savetxt(fileID, np.matrix(avgResult), fmt='%.4f', delimiter='\t')

    minResult = np.min(result, axis=0)
    fileID.write('Min:\t')
    np.savetxt(fileID, np.matrix(minResult), fmt='%.4f', delimiter='\t')

    fileID.write('\n==================================\n')
    fileID.write('Detailed results for %d epochs:\n' % result.shape[0])
    np.savetxt(fileID, result, fmt='%.4f', delimiter='\t')
    fileID.close()
