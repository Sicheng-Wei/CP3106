import sewar


def metrics(origin, recover):
    print("MSE: ",      sewar.mse(origin,recover))
    print("RMSE: ",     sewar.rmse(origin, recover))
    print("PSNR: ",     sewar.psnr(origin, recover))
    print("SSIM: ",     sewar.ssim(origin, recover))
    print("UQI: ",      sewar.uqi(origin, recover))
