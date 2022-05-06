    import numpy as np
    import matplotlib.pyplot as plt
    from ddeint import ddeint

    def f_func(x):
        return 1/(1+np.exp(-x))

    def equation(Y, t, d):
        c=18 #non-negative constant
        a=0.7 #weights of NN are initialized randomly
        b=0.3 #weights of NN are initialized randomly
        I=0 #the external bias
        x = Y(t)
        xd = Y(t-d)
        y = -c*x+a*f_func(x)+b*f_func(xd)+I
        return y

    def initial_history_func(t):
        return 1


    plt.figure(figsize=(15, 7), dpi=80);
    plt.rcParams['font.size'] = 12;
    fig, axs = plt.subplots(1, 1);
    fig.tight_layout(rect=[0, 0, 2, 2], pad=3.0);

    ts = np.linspace(0, 100, 2000);

    ys = ddeint(equation, initial_history_func, ts, fargs=(2,));
    axs.plot(ts, ys[:], color='red', linewidth=2, label='$y(t)$');
    axs.set_title('$r=2$', fontsize=20);
    axs.legend(fontsize=20);
    axs.set_ylim([-5, 5])
    axs.set_xlim([-10, 20])

    plt.tight_layout()
    plt.savefig('graph_2.png');
