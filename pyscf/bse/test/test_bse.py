#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import unittest
from pyscf import lib, gto, scf, bse #TODO: uncomment
from pyscf.gw import gw_ac

def setUpModule():
    #doi: 10.1063/5.0023168
    global gw, nstates
    mol = gto.Mole(unit='A')
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [['O',(0.0000, 0.0000, 0.0000)],
             ['H', (0.7571, 0.0000, 0.5861)],
             ['H', (-0.7571, 0.0000, 0.5861)]]
    mol.basis = 'aug-cc-pVTZ'
    mol.build()
    mf = scf.RHF(mol).run()
    
    gw = gw_ac.GWAC(mf)
    gw.kernel()
    
    nstates = 5 # make sure first 3 states are converged

def tearDownModule():
    global gw
    gw._scf.mol.stdout.close()
    gw.stdout.close()
    del gw

#TODO: update the ref values
class KnownValues(unittest.TestCase):
    def test_tda_bse_singlet(self):
        mybse = gw.TDA_BSE().set(nstates=nstates)
        e = mybse.kernel()[0]
        ref = [8.09129, 9.78553, 10.41702] #[8.104560117202942, 9.78425883863174, 10.43390150150587]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)

    def test_tda_bse_triplet(self):
        mybse = gw.TDA_BSE().set(nstates=nstates)
        mybse.singlet = False
        e = mybse.kernel()[0]
        ref = [7.61802, 9.59825, 9.79518] #[7.635794126610576, 9.607487101850701, 9.826855704109516]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)

    #lit ref is for BSE, not TDA-BSE
    def test_bse_singlet(self):
        mybse = gw.BSE().set(nstates=nstates)
        e = mybse.kernel()[0]
        ref = [8.09129, 9.78553, 10.41702] #[8.08552929,  9.78006193, 10.41286796]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)

    def test_bse_triplet(self):
        mybse = gw.BSE().set(nstates=nstates)
        mybse.singlet = False
        e = mybse.kernel()[0]
        ref = [7.61802, 9.59825, 9.79518] #[7.61167318  9.59271311  9.79013569]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)


if __name__ == "__main__":
    print("Full Tests for TDA-BSE and BSE")
    unittest.main()
