# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Authors: Sylvia Bintrim <sylviajoy1@gmail.com>
#         Tim
#

from pyscf import scf
from pyscf import dft
#TODO
#from pyscf.bse import bse
#from pyscf.tdscf.bse import BSE

def BSE(mf):
    if isinstance(mf, scf.hf.KohnShamDFT):
        raise RuntimeError('TDHF does not support DFT object %s' % mf)
    mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF):
        # Is it correct to call TDUHF for ROHF?
        mf = mf.to_uhf()
    return mf.TDHF()

def TDA_BSE(mf):
    mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF):
        if isinstance(mf, KohnShamDFT):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    return mf.TDA()
