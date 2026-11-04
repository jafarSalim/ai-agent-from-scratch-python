"""Engineering tasks with known answers."""
import math

C = "speed_of_light_m_s"


def wavelength(f):
    return dict(q=f"What is the wavelength in mm at {f} GHz?",
                plan=[("lookup", lambda o: {"name": C}), ("convert", lambda o: {"value": f, "source": "GHz", "target": "Hz"}),
                      ("calculator", lambda o: {"expression": f"{o[0]} / {o[1]} * 1000"})],
                answer=lambda o: o[2], truth=299792458.0 / (f * 1e9) * 1000)


def noise(bw):
    return dict(q=f"Thermal noise power in dBm for {bw} MHz at 290 K?",
                plan=[("lookup", lambda o: {"name": "boltzmann_j_k"}), ("convert", lambda o: {"value": bw, "source": "MHz", "target": "Hz"}),
                      ("calculator", lambda o: {"expression": f"10 * log10({o[0]} * 290 * {o[1]} * 1000)"})],
                answer=lambda o: o[2], truth=10 * math.log10(1.380649e-23 * 290 * bw * 1e6 * 1000))


def capacity(bw, snr):
    return dict(q=f"Shannon capacity in Mbit/s for {bw} MHz at {snr} dB?",
                plan=[("convert", lambda o: {"value": bw, "source": "MHz", "target": "Hz"}),
                      ("calculator", lambda o: {"expression": f"{o[0]} * log2(1 + 10 ** ({snr} / 10)) / 1000000"})],
                answer=lambda o: o[1], truth=bw * math.log2(1 + 10 ** (snr / 10)))


def cycles(ms):
    return dict(q=f"How many clock cycles does an ESP32 run in {ms} ms?",
                plan=[("lookup", lambda o: {"name": "esp32_clock_hz"}), ("calculator", lambda o: {"expression": f"{o[0]} * {ms} / 1000"})],
                answer=lambda o: o[1], truth=240e6 * ms / 1000)


TASKS = [wavelength(f) for f in (0.868, 2.4, 3.5, 5, 28, 39, 60, 140)] + \
        [noise(b) for b in (0.125, 1, 10, 20, 100, 400)] + \
        [capacity(b, s) for b, s in ((1, 0), (5, 10), (20, 15), (100, 20), (400, 25), (20, 30))] + \
        [cycles(m) for m in (1, 2, 5, 10, 100, 1000)]
