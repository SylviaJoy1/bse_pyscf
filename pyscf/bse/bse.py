#!/usr/bin/env python
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
# Ref:
# ? TODO: make sure conjugation is correct
#


# from functools import reduce
import numpy
# import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import scf
# from pyscf import ao2mo
from pyscf import symm
from pyscf.lib import logger
from pyscf.scf import hf_symm
from pyscf.data import nist
from pyscf import __config__
#from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig
from pyscf.gw.gw_ac import get_rho_response

from pyscf.gw import gw_ac
from pyscf.gw import gw_cd
from pyscf.pbc.gw import krgw_ac
from pyscf.pbc.gw import krgw_cd

#OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_rhf_get_nto_threshold', 0.3)
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)
MO_BASE = getattr(__config__, 'MO_BASE', 1)

def gen_tda_bse_operation(gw, Lpq=None, eps_inv=None, singlet=True, orbs=None, wfnsym=None):
    '''Generate function to compute A x

    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited BSE (with the TDA) wavefunction.
    '''
    mf = gw._scf
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    # assert (mo_coeff.dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
        sym_forbid = (orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym
#        x_sym = _get_x_sym_table(mf)
#        sym_forbid = x_sym != wfnsym

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')

    mf_nocc = nocc
    if orbs is not None:
        nmo = len(orbs)
        nocc = sum([x < mf_nocc for x in orbs])
        nvir = nmo - nocc
        
    e_ia = hdiag = gw.mo_energy[viridx][:nvir] - gw.mo_energy[occidx,None][:nocc,]

    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel()
    
    if Lpq is None:
        Lpq = gw.ao2mo(mo_coeff)
    if eps_inv is None:
        naux = gw.with_df.get_naoaux()
        Pi = numpy.real(get_rho_response(0.0, mo_energy, Lpq[:,:mf_nocc, mf_nocc:]))
        eps_inv = numpy.linalg.inv(numpy.eye(naux)-Pi)
        
    Loo = Lpq[:,mf_nocc-nocc:mf_nocc, mf_nocc-nocc:mf_nocc]
    Lov = Lpq[:,mf_nocc-nocc:mf_nocc, mf_nocc:mf_nocc+nvir]
    Lvv = Lpq[:,mf_nocc:mf_nocc+nvir, mf_nocc:mf_nocc+nvir]

    def vind(zs):
        zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        v1ov = numpy.einsum('xia,ia->xia', zs, e_ia)
        v1ov -= lib.einsum('Pji, PQ, Qab, xjb->xia', Loo, eps_inv, Lvv, zs)
        if singlet:
            v1ov += 2*lib.einsum('Qia, Qjb,xjb->xia', Lov, Lov, zs)

        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
        return v1ov.reshape(v1ov.shape[0],-1)

    return vind, hdiag
gen_tda_bse_hop = gen_tda_bse_operation

def as_scanner(td):
    '''Generating a scanner/solver for TDA/TDHF/TDDFT PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total TDA/TDHF/TDDFT energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    TDA/TDDFT and the underlying SCF objects (conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

        >>> from pyscf import gto, scf, tdscf
        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
        >>> td_scanner = tdscf.TDHF(scf.RHF(mol)).as_scanner()
        >>> de = td_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
        [ 0.34460866  0.34460866  0.7131453 ]
        >>> de = td_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
        [ 0.14844013  0.14844013  0.47641829]
    '''
    if isinstance(td, lib.SinglePointScanner):
        return td

    logger.info(td, 'Set %s as a scanner', td.__class__)
    name = td.__class__.__name__ + TD_Scanner.__name_mixin__
    return lib.set_class(TD_Scanner(td), (TD_Scanner, td.__class__), name)
    
class TD_Scanner(lib.SinglePointScanner):
    def __init__(self, td):
        self.__dict__.update(td.__dict__)
        self._scf = td._scf.as_scanner()

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)

        mf_scanner = self._scf
        mf_e = mf_scanner(mol)
        self.kernel(**kwargs)
        return mf_e + self.e

from pyscf.tdscf import rhf
class BSEBase(rhf.TDBase):
    conv_tol = getattr(__config__, 'tdscf_rhf_TDA_conv_tol', 1e-5) #TODO: make this smaller?
    nstates = getattr(__config__, 'tdscf_rhf_TDA_nstates', 3)
    singlet = getattr(__config__, 'tdscf_rhf_TDA_singlet', True)
    orbs = getattr(__config__, 'orbs', None)
    lindep = getattr(__config__, 'tdscf_rhf_TDA_lindep', 1e-12)
    level_shift = getattr(__config__, 'tdscf_rhf_TDA_level_shift', 0)
    max_space = getattr(__config__, 'tdscf_rhf_TDA_max_space', 50)
    max_cycle = getattr(__config__, 'tdscf_rhf_TDA_max_cycle', 100)
    # Low excitation filter to avoid numerical instability
    positive_eig_threshold = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)
    # Threshold to handle degeneracy in init guess
    deg_eia_thresh = getattr(__config__, 'tdscf_rhf_TDDFT_deg_eia_thresh', 1e-3)

    _keys = {
        'conv_tol', 'nstates', 'singlet', 'lindep', 'level_shift',
        'max_cycle', 'mol', 'chkfile', 'wfnsym', 'converged', 'e', 'xy', 'orbs'
    }

    def __init__(self, gw, orbs=None):
        self.verbose = gw.verbose
        self.stdout = gw.stdout
        self.mol = gw._scf.mol
        self._scf = gw._scf
        self.max_memory = gw.max_memory
        self.chkfile = gw._scf.chkfile

        self.wfnsym = None
        
        if not (isinstance(gw, gw_ac.GWAC) or isinstance(gw, krgw_ac.KRGWAC) or isinstance(gw, gw_cd.GWCD) or isinstance(gw, krgw_cd.KRGWCD)):
            raise NotImplementedError('Only GW-AC and GW-CD are supported for the BSE.')
        if (isinstance(gw, krgw_ac.KRGWAC) or isinstance(gw, krgw_cd.KRGWCD)):
            assert gw.kpts == [[0,0,0]]

        # xy = (X,Y), normalized to 1/2: 2(XX-YY) = 1
        # In TDA, Y = 0
        self.converged = None
        self.e = None
        self.xy = None
        self.eps_inv = None
        self.Lpq = None

    @property
    def nroots(self):
        return self.nstates
    @nroots.setter
    def nroots(self, x):
        self.nstates = x

    @property
    def e_tot(self):
        '''Excited state energies'''
        return self._scf.e_tot + self.e

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self._scf.__class__)
        if self.singlet is None:
            log.info('nstates = %d', self.nstates)
        elif self.singlet:
            log.info('nstates = %d singlet', self.nstates)
        else:
            log.info('nstates = %d triplet', self.nstates)
        if self.orbs is not None:
            log.info('orbs = {}'.format(self.orbs))
        log.info('deg_eia_thresh = %.3e', self.deg_eia_thresh)
        log.info('wfnsym = %s', self.wfnsym)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('eigh lindep = %g', self.lindep)
        log.info('eigh level_shift = %g', self.level_shift)
        log.info('eigh max_space = %d', self.max_space)
        log.info('eigh max_cycle = %d', self.max_cycle)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        if not gw.converged:
            log.warn('GW is not converged')
        log.info('\n')

    def check_sanity(self):
        if self._scf.mo_coeff is None:
            raise RuntimeError('SCF object is not initialized')
        if gw.mo_energy is None:
            raise RuntimeError('GW object is not initialized')
        lib.StreamObject.check_sanity(self)

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    def gen_vind(self, gw=None):
        raise NotImplementedError #defined for each subclass
        
    def get_ab(self, gw=None):
        raise NotImplementedError

    def get_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            return x/diagd
        return precond

    # from pyscf.tdscf.rhf import analyze, get_nto, oscillator_strength
    # oscillator_strength = oscillator_strength
    # analyze = analyze
    # get_nto = get_nto

    # from pyscf.tdscf.rhf import _contract_multipole, transition_dipole, transition_velocity_dipole,\
    # transition_magnetic_dipole, transition_quadrupole, transition_velocity_quadrupole,\
    # transition_magnetic_quadrupole, transition_octupole, transition_velocity_octupole
    # _contract_multipole = _contract_multipole  # needed by following methods
    # transition_dipole              = transition_dipole
    # transition_quadrupole          = transition_quadrupole
    # transition_octupole            = transition_octupole
    # transition_velocity_dipole     = transition_velocity_dipole
    # transition_velocity_quadrupole = transition_velocity_quadrupole
    # transition_velocity_octupole   = transition_velocity_octupole
    # transition_magnetic_dipole     = transition_magnetic_dipole
    # transition_magnetic_quadrupole = transition_magnetic_quadrupole

    as_scanner = as_scanner

#    def nuc_grad_method(self):
#        from pyscf.grad import tdrhf
#        return tdrhf.Gradients(self)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if not all(self.converged):
            logger.note(self, 'BSE states %s not converged.',
                        [i for i, x in enumerate(self.converged) if not x])
        logger.note(self, 'Excited State energies (eV)\n%s', self.e * nist.HARTREE2EV)
        return self

    def to_gpu(self):
        raise NotImplementedError

class TDA_BSE(BSEBase):
    '''Tamm-Dancoff approximation to the BSE

    Attributes:
        conv_tol : float
            Diagonalization convergence tolerance.  Default is 1e-9.
        nstates : int
            Number of BSE states to be computed. Default is 3.

    Saved results:

        converged : bool
            Diagonalization converged or not
        e : 1D array
            excitation energy for each excited state.
        xy : A list of two 2D arrays
            The two 2D arrays are Excitation coefficients X (shape [nocc,nvir])
            and de-excitation coefficients Y (shape [nocc,nvir]) for each
            excited state.  (X,Y) are normalized to 1/2 in RHF/RKS methods and
            normalized to 1 for UHF/UKS methods. In the TDA calculation, Y = 0.
    '''
    def gen_vind(self, gw=None):
        '''Generate function to compute Ax'''
        if self.Lpq is None:
            self.Lpq = gw.ao2mo(gw._scf.mo_coeff)
        if self.eps_inv is None:
            naux = gw.with_df.get_naoaux()
            Pi = numpy.real(get_rho_response(0.0, gw._scf.mo_energy, self.Lpq[:,:gw.nocc, gw.nocc:]))
            self.eps_inv = numpy.linalg.inv(numpy.eye(naux)-Pi)
        return gen_tda_bse_hop(gw, self.Lpq, self.eps_inv, singlet=self.singlet, orbs=self.orbs, wfnsym=self.wfnsym)

    def init_guess(self, mf, orbs=None, nstates=None, wfnsym=None):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym
        if orbs is None: orbs = self.orbs

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        
        mf_nocc = nocc
        if orbs is not None:
            nmo = len(orbs)
            nocc = sum([x < mf_nocc for x in orbs])
            nvir = nmo - nocc
            
        e_ia = mo_energy[viridx][:nvir] - mo_energy[occidx,None][:nocc,]

        if wfnsym is not None and mf.mol.symmetry:
            if isinstance(wfnsym, str):
                wfnsym = symm.irrep_name2id(mf.mol.groupname, wfnsym)
            wfnsym = wfnsym % 10  # convert to D2h subgroup
            orbsym = hf_symm.get_orbsym(mf.mol, mf.mo_coeff)
            orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
            # sym_forbid = (orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym
#            orbsym = hf_symm.get_orbsym(mf.mol, mf.mo_coeff)
#            orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
#            e_ia[(orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym] = 1e99

        nov = e_ia.size
        nstates = min(nstates, nov)
        e_ia = e_ia.ravel()
        e_threshold = numpy.sort(e_ia)[nstates-1]
        e_threshold += self.deg_eia_thresh

        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations
        return x0

    def kernel(self, x0=None, nstates=None, orbs=None):
        '''TDA-BSE diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        if orbs is None:
            orbs = self.orbs

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(gw)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        if x0 is None:
            x0 = self.init_guess(gw._scf, self.orbs, self.nstates)

        self.converged, self.e, x1 = \
                lib.davidson1(vind, x0, precond,
                              tol=self.conv_tol,
                              nroots=nstates, lindep=self.lindep,
                              max_cycle=self.max_cycle,
                              max_space=self.max_space, pick=pickeig,
                              verbose=log)

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        # 1/sqrt(2) because self.x is for alpha excitation and 2(X^+*X) = 1
        mf_nocc = nocc
        mf_nvir = nvir
        if orbs is not None:
            nmo = len(orbs)
            nocc = sum([x < mf_nocc for x in orbs])
            nvir = nmo - nocc
        #It may be best to pad the xy with 0's here for post-processing
        #TODO: test whether sym, analyze results with and without orbs are the same
        self.xy = [(numpy.pad(xi.reshape(nocc,nvir)*numpy.sqrt(.5), ((mf_nocc - nocc,0),(0,mf_nvir - nvir)), mode='constant', constant_values=0),0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA', *cpu0)
        self._finalize()
        return self.e, self.xy

    to_gpu = lib.to_gpu

#TODO: check the equations and add orbs
# def gen_bse_operation(gw, singlet=True, wfnsym=None):
#     '''Generate function to compute

#     [ A  B][X]
#     [-B -A][Y]
#     '''
#     mf = gw._scf
#     mol = mf.mol
#     mo_coeff = mf.mo_coeff
#     # assert (mo_coeff.dtype == numpy.double)
#     mo_energy = mf.mo_energy
#     qp_energy = gw.mo_energy
#     mo_occ = mf.mo_occ
#     nao, nmo = mo_coeff.shape
#     occidx = numpy.where(mo_occ==2)[0]
#     viridx = numpy.where(mo_occ==0)[0]
#     nocc = len(occidx)
#     nvir = len(viridx)
#     orbv = mo_coeff[:,viridx]
#     orbo = mo_coeff[:,occidx]

#     if wfnsym is not None and mol.symmetry:
#         if isinstance(wfnsym, str):
#             wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
#         wfnsym = wfnsym % 10  # convert to D2h subgroup
#         orbsym = hf_symm.get_orbsym(mol, mo_coeff)
#         orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
#         sym_forbid = (orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym
#         # sym_forbid = _get_x_sym_table(mf) != wfnsym

#     e_ia = hdiag = qp_energy[viridx] - qp_energy[occidx,None]
#     if wfnsym is not None and mol.symmetry:
#         hdiag[sym_forbid] = 0
#     hdiag = numpy.hstack((hdiag.ravel(), -hdiag.ravel()))

#     mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')

#     def vind(xys):
#         xys = numpy.asarray(xys).reshape(-1,2,nocc,nvir)
#         if wfnsym is not None and mol.symmetry:
#             # shape(nz,2,nocc,nvir): 2 ~ X,Y
#             xys = numpy.copy(xys)
#             xys[:,:,sym_forbid] = 0

#         xs, ys = xys.transpose(1,0,2,3)
        
#         #TODO: work out this
# #        # *2 for double occupancy
# #        dms  = lib.einsum('xov,qv,po->xpq', xs*2, orbv.conj(), orbo)
# #        dms += lib.einsum('xov,pv,qo->xpq', ys*2, orbv, orbo.conj())
# #        v1ao = vresp(dms) # = <mb||nj> Xjb + <mj||nb> Yjb
# #        # A ~= <ib||aj>, B = <ij||ab>
# #        # AX + BY
# #        # = <ib||aj> Xjb + <ij||ab> Yjb
# #        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cmi* Cna
# #        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
# #        # (B*)X + (A*)Y
# #        # = <ab||ij> Xjb + <aj||ib> Yjb
# #        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cma* Cni
# #        v1vo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())
# #        v1ov += numpy.einsum('xia,ia->xia', xs, e_ia)  # AX
# #        v1vo += numpy.einsum('xia,ia->xia', ys, e_ia.conj())  # (A*)Y

#         if wfnsym is not None and mol.symmetry:
#             v1ov[:,sym_forbid] = 0
#             v1vo[:,sym_forbid] = 0

#         # (AX, -AY)
#         nz = xys.shape[0]
#         hx = numpy.hstack((v1ov.reshape(nz,-1), -v1vo.reshape(nz,-1)))
#         return hx

#     return vind, hdiag


# class BSE(BSEBase):
#     '''Bethe-Salpeter equation

#     Attributes:
#         conv_tol : float
#             Diagonalization convergence tolerance.  Default is 1e-9.
#         nstates : int
#             Number of BSE states to be computed. Default is 3.

#     Saved results:

#         converged : bool
#             Diagonalization converged or not
#         e : 1D array
#             excitation energy for each excited state.
#         xy : A list of two 2D arrays
#             The two 2D arrays are Excitation coefficients X (shape [nocc,nvir])
#             and de-excitation coefficients Y (shape [nocc,nvir]) for each
#             excited state.  (X,Y) are normalized to 1/2 in RHF/RKS methods and
#             normalized to 1 for UHF/UKS methods. In the TDA calculation, Y = 0.
#     '''
#     @lib.with_doc(gen_bse_operation.__doc__)
#     def gen_vind(self, gw=None):
#         return gen_bse_operation(gw, singlet=self.singlet, wfnsym=self.wfnsym)


#     def init_guess(self, mf, nstates=None, wfnsym=None):
#         x0 = TDA_BSE.init_guess(self, mf, nstates, wfnsym)
#         y0 = numpy.zeros_like(x0)
#         return numpy.asarray(numpy.block([[x0, y0], [y0, x0.conj()]]))
# #    def init_guess(self, gw, nstates=None, wfnsym=None, return_symmetry=False):
# #        if return_symmetry:
# #            x0, x0sym = TDA_BSE.init_guess(self, gw, nstates, wfnsym, return_symmetry)
# #            y0 = numpy.zeros_like(x0)
# #            return numpy.hstack([x0, y0]), x0sym
# #        else:
# #            x0 = TDA_BSE.init_guess(self, gw, nstates, wfnsym, return_symmetry)
# #            y0 = numpy.zeros_like(x0)
# #            return numpy.hstack([x0, y0])
            
#     def kernel(self, x0=None, nstates=None):
#         '''TDHF diagonalization with non-Hermitian eigenvalue solver
#         '''
#         cpu0 = (logger.process_clock(), logger.perf_counter())
#         self.check_sanity()
#         self.dump_flags()
#         if nstates is None:
#             nstates = self.nstates
#         else:
#             self.nstates = nstates
#         mol = self.mol

#         log = logger.Logger(self.stdout, self.verbose)

#         vind, hdiag = self.gen_vind(gw)
#         precond = self.get_precond(hdiag)

#         # handle single kpt PBC SCF
#         if getattr(gw._scf, 'kpt', None) is not None:
#             from pyscf.pbc.lib.kpts_helper import gamma_point
#             real_system = (gamma_point(gw._scf.kpt) and
#                            gw._scf.mo_coeff[0].dtype == numpy.double)
#         else:
#             real_system = True

#         # We only need positive eigenvalues
#         def pickeig(w, v, nroots, envs):
#             realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
#                                   (w.real > self.positive_eig_threshold))[0]
#             # If the complex eigenvalue has small imaginary part, both the
#             # real part and the imaginary part of the eigenvector can
#             # approximately be used as the "real" eigen solutions.
#             return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, real_system)


#         if x0 is None:
#             x0 = self.init_guess(self._scf, self.nstates)

#         self.converged, w, x1 = \
#                 lib.davidson_nosym1(vind, x0, precond,
#                                     tol=self.conv_tol,
#                                     nroots=nstates, lindep=self.lindep,
#                                     max_cycle=self.max_cycle,
#                                     max_space=self.max_space, pick=pickeig,
#                                     verbose=log)
# #        x0sym = None
# #        if x0 is None:
# #            x0, x0sym = self.init_guess(
# #                self.gw, self.nstates, return_symmetry=True)
# #        elif mol.symmetry:
# #            x_sym = y_sym = _get_x_sym_table(self._scf).ravel()
# #            x_sym = numpy.append(x_sym, y_sym)
# #            x0sym = [_guess_wfnsym_id(self, x_sym, x) for x in x0]
# #
# #        self.converged, w, x1 = lr_eig(
# #            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
# #            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
# #            max_memory=self.max_memory, verbose=log)

#         #TODO: update for orbs
#         nocc = (self._scf.mo_occ>0).sum()
#         nmo = self._scf.mo_occ.size
#         nvir = nmo - nocc
#         self.e = w
#         def norm_xy(z):
#             x, y = z.reshape(2,nocc,nvir)
#             norm = lib.norm(x)**2 - lib.norm(y)**2
#             norm = numpy.sqrt(.5/norm)  # normalize to 0.5 for alpha spin
#             return x*norm, y*norm
#         self.xy = [norm_xy(z) for z in x1]

#         if self.chkfile:
#             lib.chkfile.save(self.chkfile, 'bse/e', self.e)
#             lib.chkfile.save(self.chkfile, 'bse/xy', self.xy)

#         log.timer('BSE', *cpu0)
#         self._finalize()
#         return self.e, self.xy

# #    def nuc_grad_method(self):
# #        from pyscf.grad import tdrhf
# #        return tdrhf.Gradients(self)

#     to_gpu = lib.to_gpu

from pyscf import gw
gw.gw_ac.GWAC.TDA_BSE = lib.class_as_method(TDA_BSE)
# gw.gw_ac.GWAC.BSE = lib.class_as_method(BSE)
gw.gw_cd.GWCD.TDA_BSE = lib.class_as_method(TDA_BSE)
# gw.gw_cd.GWCD.BSE = lib.class_as_method(BSE)

# del (OUTPUT_THRESHOLD)


if __name__ == "__main__":
    from pyscf import lib, bse

    mol = gto.Mole(unit='A')
    mol.verbose = 9
    mol.output = '/dev/null'
    mol.atom = [['O',(0.0000, 0.0000, 0.0000)],
             ['H', (0.7571, 0.0000, 0.5861)],
             ['H', (-0.7571, 0.0000, 0.5861)]]
    mol.basis = 'aug-cc-pVTZ'
    mol.symmetry = True
    mol.build()
    mf = scf.RHF(mol).run()
    
    gw = gw_ac.GWAC(mf)
    gw.kernel()
    nmo = gw.nmo
    
    nstates = 5 # make sure first 3 states are converged

    def test_tda_bse_singlet():
        mybse = gw.TDA_BSE().set(nstates=nstates)
        mybse.orbs = range(nmo)
        e = mybse.kernel()[0]
        #TODO: why does analyze not work?
        mybse.analyze()
        print(e*27.2114)
        ref = [8.10895514,  9.78889175, 10.43696308] # my original TDA-BSE [8.104560117202942, 9.78425883863174, 10.43390150150587] # lit [8.09129, 9.78553, 10.41702]
        numpy.testing.assert_almost_equal(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)

    def test_tda_bse_triplet():
        mybse = gw.TDA_BSE().set(nstates=nstates)
        mybse.singlet = False
        e = mybse.kernel()[0]
        print(e*27.2114)
        ref = [7.64039948,  9.61227414,  9.83007805] # my original TDA-BSE [7.635794126610576, 9.607487101850701, 9.826855704109516] # lit [7.61802, 9.59825, 9.79518]
        numpy.testing.assert_almost_equal(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)

    #lit ref is for BSE, not TDA-BSE
    def test_bse_singlet():
        mybse = gw.BSE().set(nstates=nstates)
        e = mybse.kernel()[0]
        ref = [8.09129, 9.78553, 10.41702] #[8.08552929,  9.78006193, 10.41286796]
        numpy.testing.assert_almost_equal(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)

    def test_bse_triplet():
        mybse = gw.BSE().set(nstates=nstates)
        mybse.singlet = False
        e = mybse.kernel()[0]
        ref = [7.61802, 9.59825, 9.79518] #[7.61167318  9.59271311  9.79013569]
        numpy.testing.assert_almost_equal(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)
        
    test_tda_bse_singlet()
    test_tda_bse_triplet()
    # test_bse_singlet()
    # test_bse_triplet()