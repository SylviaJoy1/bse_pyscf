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

#from pyscf import scf
#from pyscf import dft
from pyscf import gw
from pyscf.bse import BSE
from pyscf.bse import TDA_BSE

def BSE(gw):
    return gw.BSE()

def TDA_BSE(gw):
    return gw.TDA_BSE()
