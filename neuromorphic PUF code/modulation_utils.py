import numpy as np
import abc
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class Modem(metaclass=abc.ABCMeta):
    """
    Abstract base class for digital modulation schemes.
    Handles general modulate/demodulate logic shared by PAM, PSK, QAM, FSK.

    Attributes:
        M (int): Number of symbols in the modulation constellation.
        name (str): Modulation scheme type (e.g., 'QAM', 'PSK').
        constellation (np.ndarray): Complex reference points for modulation.
        coherence (str): Only relevant for FSK – 'coherent' or 'noncoherent'.
    """
    def __init__(self, M, constellation, name, coherence=None):
        if (M < 2) or ((M & (M - 1)) != 0):
            raise ValueError('M should be a power of 2')

        # Validate coherence mode if FSK is used
        if name.lower() == 'fsk':
            if coherence.lower() in ['coherent', 'noncoherent']:
                self.coherence = coherence
            else:
                raise ValueError('Coherence must be "coherent" or "noncoherent"')
        else:
            self.coherence = None

        self.M = M
        self.name = name
        self.constellation = constellation

    def plotConstellation(self):
        """
        Plot the complex I/Q constellation diagram for the modem.
        FSK is excluded due to its higher-dimensional representation.
        """
        from math import log2
        if self.name.lower() == 'fsk':
            return 0

        fig, axs = plt.subplots(1, 1)
        axs.plot(np.real(self.constellation), np.imag(self.constellation), 'o')

        for i in range(self.M):
            axs.annotate("{0:0{1}b}".format(i, int(log2(self.M))),
                         (np.real(self.constellation[i]), np.imag(self.constellation[i])))

        axs.set_title('Constellation')
        axs.set_xlabel('I')
        axs.set_ylabel('Q')
        plt.grid(True)
        plt.show()

    def modulate(self, inputSymbols):
        """
        Map input symbol indices (0 to M-1) to complex constellation points.

        Parameters:
            inputSymbols (list or ndarray): Integer-valued symbols.

        Returns:
            ndarray: Modulated complex signal.
        """
        if isinstance(inputSymbols, list):
            inputSymbols = np.array(inputSymbols)

        if not (0 <= inputSymbols.all() <= self.M - 1):
            raise ValueError('Values for inputSymbols are beyond the range 0 to M-1')

        return self.constellation[inputSymbols]

    def demodulate(self, receivedSyms):
        """
        Map received complex symbols to closest constellation points.

        Parameters:
            receivedSyms (list or ndarray): Received signal values.

        Returns:
            ndarray: Symbol indices after detection.
        """
        if isinstance(receivedSyms, list):
            receivedSyms = np.array(receivedSyms)

        return self.iqDetector(receivedSyms)

    def iqDetector(self, receivedSyms):
        """
        Use Euclidean distance in I/Q plane to determine closest constellation point.

        Parameters:
            receivedSyms (ndarray): Received symbols (complex).

        Returns:
            ndarray: Detected symbol indices.
        """
        XA = np.column_stack((np.real(receivedSyms), np.imag(receivedSyms)))
        XB = np.column_stack((np.real(self.constellation), np.imag(self.constellation)))

        d = cdist(XA, XB, metric='euclidean')  # Compute distances
        return np.argmin(d, axis=1)  # Return closest symbol index


class PAMModem(Modem):
    """
    Pulse Amplitude Modulation (PAM) modem.
    Derived from the base Modem class.
    """
    def __init__(self, M):
        m = np.arange(0, M)
        constellation = 2 * m + 1 - M + 1j * 0  # Symmetric 1D constellation
        super().__init__(M, constellation, name='PAM')


class PSKModem(Modem):
    """
    Phase Shift Keying (PSK) modem using normalized circular constellation.
    """
    def __init__(self, M):
        m = np.arange(0, M)
        I = 1 / np.sqrt(2) * np.cos(m / M * 2 * np.pi)
        Q = 1 / np.sqrt(2) * np.sin(m / M * 2 * np.pi)
        constellation = I + 1j * Q
        super().__init__(M, constellation, name='PSK')


class QAMModem(Modem):
    """
    Quadrature Amplitude Modulation (QAM) modem using square constellations.
    Gray-coded mapping is used.
    """
    def __init__(self, M):
        if (M == 1) or (np.mod(np.log2(M), 2) != 0):
            raise ValueError('Only square MQAM supported. M must be an even power of 2')

        n = np.arange(0, M)
        a = np.asarray([x ^ (x >> 1) for x in n])  # Gray code mapping
        D = np.sqrt(M).astype(int)
        a = np.reshape(a, (D, D))

        # Flip every other row for proper Gray code walk
        oddRows = np.arange(start=1, stop=D, step=2)
        a[oddRows, :] = np.fliplr(a[oddRows, :])
        nGray = np.reshape(a, (M))

        # Map to 2D I/Q constellation
        x, y = np.divmod(nGray, D)
        Ax = 2 * x + 1 - D
        Ay = 2 * y + 1 - D
        constellation = Ax + 1j * Ay

        super().__init__(M, constellation, name='QAM')


class FSKModem(Modem):
    """
    Frequency Shift Keying (FSK) modem.
    Modulation is defined by diagonal unitary matrix for each symbol.
    """
    def __init__(self, M, coherence='coherent'):
        if coherence.lower() == 'coherent':
            phi = np.zeros(M)  # Phase = 0 for coherent FSK
        elif coherence.lower() == 'noncoherent':
            phi = 2 * np.pi * np.random.rand(M)  # Random phases
        else:
            raise ValueError('Coherence must be "coherent" or "noncoherent"')

        # Use diagonal phase matrix
        constellation = np.diag(np.exp(1j * phi))
        super().__init__(M, constellation, name='FSK', coherence=coherence.lower())

    def demodulate(self, receivedSyms, coherence='coherent'):
        """
        Override demodulation for FSK depending on coherence type.
        """
        if coherence.lower() == 'coherent':
            return self.iqDetector(receivedSyms)
        elif coherence.lower() == 'noncoherent':
            return np.argmax(np.abs(receivedSyms), axis=1)
        else:
            raise ValueError('Coherence must be "coherent" or "noncoherent"')
