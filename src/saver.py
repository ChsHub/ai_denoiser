from logging import info
from threading import Thread
from time import sleep, perf_counter_ns


class Saver(Thread):
    def __init__(self, net):
        Thread.__init__(self, daemon=True)
        self._net = net
        self.running_loss = 0.0
        self.counter = 0
        self.epoch = 0
        self.timer = None

    def run(self):
        """
        Overwrite Thread.run().
        Periodically save the neural net state
        """
        info('RUNNING')
        while True:
            sleep(300)
            self._net.save_state(running_loss=self.running_loss / self.counter, epoch=self.epoch)
            info('STATE SAVED // RUNNING LOSS: %.4f AVG BATCH TIME: %i ms' %
                 (self.running_loss / self.counter, (perf_counter_ns() - self.timer._start) / self.counter / 1_000_000))

