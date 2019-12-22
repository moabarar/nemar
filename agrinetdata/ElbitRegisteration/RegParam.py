from pathlib import Path

import cv2
import numpy as np

from agrinetdata.ElbitRegisteration.Constants import *


class RegParam:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self.T = {}
        self.b_x = {}
        self.b_y = {}
        for modal in ALL_MODALS:
            self.T[modal] = np.eye(2, 3, dtype=self.dtype)
            self.b_x[modal] = (0.0, 0.0)
            self.b_y[modal] = (0.0, 0.0)

    def set_transformation(self, modal, transformation):
        if transformation is None or transformation.shape[0] != 2 or transformation.shape[1] != 3:
            return False
        self.T[modal] = np.copy(transformation)
        return True

    def set_bx(self, modal, bx):
        assert isinstance(bx, tuple)
        assert modal in self.b_x.keys()
        if len(bx) != 2:
            return False
        self.b_x[modal] = bx
        return True

    def set_by(self, modal, by):
        assert isinstance(by, tuple)
        assert modal in self.b_y.keys()
        if len(by) != 2:
            return False
        self.b_y[modal] = by
        return True

    def init_from_xml_file(self, fpath):
        fs = cv2.FileStorage(fpath, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            print('Couldn\'t open calibration file')
            return None
        root_node = fs.getFirstTopLevelNode()
        for idx, modal in enumerate(ALL_MODALS):
            modal_node = root_node.getNode(modal)
            if modal_node.isNone():
                print('{} : Modal {} RegParam not found'.format(fpath, modal))
            T = modal_node.getNode('Tform').mat()
            if not self.set_transformation(modal, T):
                return None
            bn = modal_node.getNode('bx')
            bx = (bn.at(0).real(), bn.at(1).real())
            if not self.set_bx(modal, bx):
                return None
            bn = modal_node.getNode('by')
            by = (bn.at(0).real(), bn.at(1).real())
            if not self.set_by(modal, by):
                return None
        fs.release()
        return self

    def init_from_txt_file(self, fpath):
        with open(fpath, mode='r') as f:
            if f.mode == 'r':
                words = f.read().split(' ')
                i = 0
                idx = 0
                while i < len(words):
                    if words[i] == 'Tform':
                        i += 1
                        T = np.zeros((2, 3), self.dtype)
                        for l in range(2):
                            for k in range(3):
                                T[l, k] = float(words[i])
                                i += 1
                        if not self.set_transformation(ALL_MODALS[idx], T):
                            return None
                        assert words[i] == 'Line'
                        i += 1
                        if not self.set_bx(ALL_MODALS[idx], (float(words[i]), float(words[i + 1]))):
                            return None
                        i += 2
                        if not self.set_by(ALL_MODALS[idx], (float(words[i]), float(words[i + 1]))):
                            return None
                        i += 2
                        idx += 1
                    else:
                        i += 1
                f.close()
                return self
            else:
                print('Couldn\'t open calibration file')
                return None

    def apply_dist_fix(self, new_dist):
        alpha = 1.0 / new_dist
        for modal in ALL_MODALS:
            self.T[modal][1, 2] -= (self.b_y[modal][0] + self.b_y[modal][1] * alpha)
            self.T[modal][0, 2] -= self.b_x[modal][0] + self.b_x[modal][1] * alpha


if __name__ == '__main__':
    p = Path('../test/Calib.txt')
    print(p.absolute())
    param = RegParam().init_from_file(p.absolute())
    print(param.T['RGBD'])
