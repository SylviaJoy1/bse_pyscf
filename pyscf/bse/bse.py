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
# ?
#


from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import symm
from pyscf.lib import logger
from pyscf.scf import hf_symm
from pyscf.scf import _response_functions
from pyscf.data import nist
from pyscf import __config__
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig
from pyscf import gw

#OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_rhf_get_nto_threshold', 0.3)
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)
MO_BASE = getattr(__config__, 'MO_BASE', 1)

#TODO: throw some error if not GW AC/CD

#TODO: add gw frozen, bse orbs
def gen_tda_bse_operation(gw, singlet=True, wfnsym=None):
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
    qp_energy = gw.mo_energy
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
        x_sym = _get_x_sym_table(mf)
        sym_forbid = x_sym != wfnsym

    #TODO: should this be mo_energy instead?
    e_ia = hdiag = qp_energy[viridx] - qp_energy[occidx,None]

    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel()

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')

    #TODO: define this stuff
#    nmo = len(orbs)
#    nocc = sum([x < bse.mf_nocc for x in orbs])
#    nvir = nmo - nocc
    qp_e_occ = qp_energy[:nocc]#[bse.mf_nocc-nocc:bse.mf_nocc]
    qp_e_vir = qp_energy[nocc:]#[bse.mf_nocc:bse.mf_nocc+nvir]
    i_tilde =
    eris =  eris.Lov etc

    def vind(zs):
        zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        v1ov = np.einsum('xia,ia->xia', zs, e_ia) #where e_ia is qp
        v1ov -= lib.einsum('Pji, PQ, Qab, xjb->xia', eris.Loo, i_tilde, eris.Lvv, zs)
        if bse.singlet:
            v1ov += 2*lib.einsum('Qia, Qjb,xjb->xia', eris.Lov, eris.Lov, zs)

        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
        return v1ov.reshape(v1ov.shape[0],-1)

    return vind, hdiag
gen_tda_bse_hop = gen_tda_bse_operation

#TODO: restrict to just orbs segments
def _get_x_sym_table(mf):
    '''Irrep (up to D2h symmetry) of each coefficient in X[nocc,nvir]'''
    mol = mf.mol
    mo_occ = mf.mo_occ
    orbsym = hf_symm.get_orbsym(mol, mf.mo_coeff)
    orbsym = orbsym % 10  # convert to D2h irreps
    return orbsym[mo_occ==2,None] ^ orbsym[mo_occ==0]

#TODO: re-implement get_nto() and analyze() and _analyze_wfnsym() and _guess_wfnsym_id()
# if necessary to take into account orbs subspace
#otherwise, just import tdscf functions

#bseobj should have attribute orbs
#TODO: include orbs here
def _contract_multipole(tdobj, ints, hermi=True, xy=None):
    '''ints is the integral tensor of a spin-independent operator'''
    if xy is None: xy = tdobj.xy
    nstates = len(xy)
    pol_shape = ints.shape[:-2]
    nao = ints.shape[-1]

    if not tdobj.singlet:
        return numpy.zeros((nstates,) + pol_shape)

    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    orbo = mo_coeff[:,mo_occ==2]
    orbv = mo_coeff[:,mo_occ==0]

    #Incompatible to old numpy version
    #ints = numpy.einsum('...pq,pi,qj->...ij', ints, orbo.conj(), orbv)
    ints = lib.einsum('xpq,pi,qj->xij', ints.reshape(-1,nao,nao), orbo.conj(), orbv)
    pol = numpy.array([numpy.einsum('xij,ij->x', ints, x) * 2 for x,y in xy])
    if isinstance(xy[0][1], numpy.ndarray):
        if hermi:
            pol += [numpy.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
        else:  # anti-Hermitian
            pol -= [numpy.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
    pol = pol.reshape((nstates,)+pol_shape)
    return pol

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

#TODO: check this
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


class BSEBase(lib.StreamObject):
    conv_tol = getattr(__config__, 'tdscf_rhf_TDA_conv_tol', 1e-5) #TODO: make this smaller?
    nstates = getattr(__config__, 'tdscf_rhf_TDA_nstates', 3)
    singlet = getattr(__config__, 'tdscf_rhf_TDA_singlet', True)
    lindep = getattr(__config__, 'tdscf_rhf_TDA_lindep', 1e-12)
    level_shift = getattr(__config__, 'tdscf_rhf_TDA_level_shift', 0)
#    max_space = getattr(__config__, 'tdscf_rhf_TDA_max_space', 50)
    max_cycle = getattr(__config__, 'tdscf_rhf_TDA_max_cycle', 100)
    # Low excitation filter to avoid numerical instability
    positive_eig_threshold = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)
    # Threshold to handle degeneracy in init guess
    deg_eia_thresh = getattr(__config__, 'tdscf_rhf_TDDFT_deg_eia_thresh', 1e-3)

    _keys = {
        'conv_tol', 'nstates', 'singlet', 'lindep', 'level_shift',
        'max_cycle', 'mol', 'chkfile', 'wfnsym', 'converged', 'e', 'xy', 'orbs'
    }

    def __init__(self, gw):
        self.verbose = gw.verbose
        self.stdout = gw.stdout
        self.mol = gw._scf.mol
        self._scf = gw._scf
        self.max_memory = gw.max_memory
        self.chkfile = gw.chkfile #TODO: or mf?

        self.wfnsym = None

        # xy = (X,Y), normalized to 1/2: 2(XX-YY) = 1
        # In TDA, Y = 0
        self.converged = None
        self.e = None
        self.xy = None

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
        log.info('orbs = {}'.format(self.orbs))
        log.info('deg_eia_thresh = %.3e', self.deg_eia_thresh)
        log.info('wfnsym = %s', self.wfnsym)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('eigh lindep = %g', self.lindep)
        log.info('eigh level_shift = %g', self.level_shift)
#        log.info('eigh max_space = %d', self.max_space)
        log.info('eigh max_cycle = %d', self.max_cycle)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        if not self.gw.converged:
            log.warn('GW is not converged')
        log.info('\n')

    def check_sanity(self):
        if self._scf.mo_coeff is None:
            raise RuntimeError('SCF object is not initialized')
        if self.gw.mo_energy is None:
            raise RuntimeError('GW object is not initialized')
        lib.StreamObject.check_sanity(self)

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    #TODO: did I define gen_vind with gw as arg?
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

#    analyze = analyze
#    get_nto = get_nto
    from pyscf.tdscf.rhf import oscillator_strength
    oscillator_strength = oscillator_strength

    #_contract_multipole should be our BSE one, not the tdscf one.
    _contract_multipole = _contract_multipole  # needed by following methods
    
    #make sure these imported tdscf functions use our _contract_multipole function
    from pyscf.tdscf.rhf import transition_dipole, transition_velocity_dipole,\
    transition_magnetic_dipole, transition_quadrupole, transition_velocity_quadrupole,\
    transition_magnetic_quadrupole, transition_octupole, transition_velocity_octupole
    transition_dipole              = transition_dipole
    transition_quadrupole          = transition_quadrupole
    transition_octupole            = transition_octupole
    transition_velocity_dipole     = transition_velocity_dipole
    transition_velocity_quadrupole = transition_velocity_quadrupole
    transition_velocity_octupole   = transition_velocity_octupole
    transition_magnetic_dipole     = transition_magnetic_dipole
    transition_magnetic_quadrupole = transition_magnetic_quadrupole

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
        if gw is None:
            gw = self.gw
        return gen_tda_bse_hop(gw, singlet=self.singlet, wfnsym=self.wfnsym)

    #TODO: should init guess be GW or MOs?
    #TODO: add orbs
    def init_guess(self, gw, nstates=None, wfnsym=None, return_symmetry=False):
        '''
        Generate initial guess for TDA-BSE

        Kwargs:
            nstates : int
                The number of initial guess vectors.
            wfnsym : int or str
                The irrep label or ID of the wavefunction.
            return_symmetry : bool
                Whether to return symmetry labels for initial guess vectors.
        '''
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mf = gw._scf
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        e_ia = (mo_energy[viridx] - mo_energy[occidx,None]).ravel()
        nov = e_ia.size
        nstates = min(nstates, nov)

        if (wfnsym is not None or return_symmetry) and mf.mol.symmetry:
            x_sym = _get_x_sym_table(mf).ravel()
            if wfnsym is not None:
                if isinstance(wfnsym, str):
                    wfnsym = symm.irrep_name2id(mf.mol.groupname, wfnsym)
                wfnsym = wfnsym % 10  # convert to D2h subgroup
                e_ia[x_sym != wfnsym] = 1e99
                nov_allowed = numpy.count_nonzero(x_sym == wfnsym)
                nstates = min(nstates, nov_allowed)

        # Find the nstates-th lowest energy gap
        e_threshold = numpy.partition(e_ia, nstates-1)[nstates-1]
        e_threshold += self.deg_eia_thresh

        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations

        if return_symmetry:
            if mf.mol.symmetry:
                x0sym = x_sym[idx]
            else:
                x0sym = None
            return x0, x0sym
        else:
            return x0

    def kernel(self, x0=None, nstates=None):
        '''TDA-BSE diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self.gw)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        x0sym = None
        if x0 is None:
            x0, x0sym = self.init_guess(
                self.gw, self.nstates, return_symmetry=True)
        elif mol.symmetry:
            x_sym = _get_x_sym_table(self._scf).ravel()
            x0sym = [_guess_wfnsym_id(self, x_sym, x) for x in x0]

        self.converged, self.e, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        # 1/sqrt(2) because self.x is for alpha excitation and 2(X^+*X) = 1
        self.xy = [(xi.reshape(nocc,nvir)*numpy.sqrt(.5),0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA', *cpu0)
        self._finalize()
        return self.e, self.xy

    to_gpu = lib.to_gpu

def gen_bse_operation(gw, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A  B][X]
    [-B -A][Y]
    '''
    mf = gw._scf
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    # assert (mo_coeff.dtype == numpy.double)
    mo_energy = mf.mo_energy
    qp_energy = gw.mo_energy
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
        sym_forbid = _get_x_sym_table(mf) != wfnsym

    e_ia = hdiag = qp_energy[viridx] - qp_energy[occidx,None]
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag.ravel(), -hdiag.ravel()))

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')

    def vind(xys):
        xys = numpy.asarray(xys).reshape(-1,2,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,nocc,nvir): 2 ~ X,Y
            xys = numpy.copy(xys)
            xys[:,:,sym_forbid] = 0

        xs, ys = xys.transpose(1,0,2,3)
        
        #TODO: work out this
#        # *2 for double occupancy
#        dms  = lib.einsum('xov,qv,po->xpq', xs*2, orbv.conj(), orbo)
#        dms += lib.einsum('xov,pv,qo->xpq', ys*2, orbv, orbo.conj())
#        v1ao = vresp(dms) # = <mb||nj> Xjb + <mj||nb> Yjb
#        # A ~= <ib||aj>, B = <ij||ab>
#        # AX + BY
#        # = <ib||aj> Xjb + <ij||ab> Yjb
#        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cmi* Cna
#        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
#        # (B*)X + (A*)Y
#        # = <ab||ij> Xjb + <aj||ib> Yjb
#        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cma* Cni
#        v1vo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())
#        v1ov += numpy.einsum('xia,ia->xia', xs, e_ia)  # AX
#        v1vo += numpy.einsum('xia,ia->xia', ys, e_ia.conj())  # (A*)Y

        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
            v1vo[:,sym_forbid] = 0

        # (AX, -AY)
        nz = xys.shape[0]
        hx = numpy.hstack((v1ov.reshape(nz,-1), -v1vo.reshape(nz,-1)))
        return hx

    return vind, hdiag


class BSE(BSEBase):
    '''Bethe-Salpeter equation

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
    @lib.with_doc(gen_bse_operation.__doc__)
    def gen_vind(self, gw=None):
        if gw is None:
            gw = self.gw
        return gen_bse_operation(gw, singlet=self.singlet, wfnsym=self.wfnsym)

    def init_guess(self, gw, nstates=None, wfnsym=None):
        if return_symmetry:
            x0, x0sym = TDA_BSE.init_guess(self, gw, nstates, wfnsym, return_symmetry)
            y0 = numpy.zeros_like(x0)
            return numpy.hstack([x0, y0]), x0sym
        else:
            x0 = TDA_BSE.init_guess(self, gw, nstates, wfnsym, return_symmetry)
            y0 = numpy.zeros_like(x0)
            return numpy.hstack([x0, y0])
            
    def kernel(self, x0=None, nstates=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
         mol = self.mol

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self.gw)
        precond = self.get_precond(hdiag)

        # handle single kpt PBC SCF
        if getattr(self.gw._scf, 'kpt', None) is not None:
            from pyscf.pbc.lib.kpts_helper import gamma_point
            real_system = (gamma_point(self.gw._scf.kpt) and
                           self.gw._scf.mo_coeff[0].dtype == numpy.double)
        else:
            real_system = True

        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            # If the complex eigenvalue has small imaginary part, both the
            # real part and the imaginary part of the eigenvector can
            # approximately be used as the "real" eigen solutions.
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, real_system)

        x0sym = None
        if x0 is None:
            x0, x0sym = self.init_guess(
                self.gw, self.nstates, return_symmetry=True)
        elif mol.symmetry:
            x_sym = y_sym = _get_x_sym_table(self._scf).ravel()
            x_sym = numpy.append(x_sym, y_sym)
            x0sym = [_guess_wfnsym_id(self, x_sym, x) for x in x0]

        self.converged, w, x1 = lr_eig(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        #TODO: update for orbs
        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        self.e = w
        def norm_xy(z):
            x, y = z.reshape(2,nocc,nvir)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = numpy.sqrt(.5/norm)  # normalize to 0.5 for alpha spin
            return x*norm, y*norm
        self.xy = [norm_xy(z) for z in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'bse/e', self.e)
            lib.chkfile.save(self.chkfile, 'bse/xy', self.xy)

        log.timer('BSE', *cpu0)
        self._finalize()
        return self.e, self.xy

#    def nuc_grad_method(self):
#        from pyscf.grad import tdrhf
#        return tdrhf.Gradients(self)

    to_gpu = lib.to_gpu

gw.gw_ac.TDA_BSE = lib.class_as_method(TDA_BSE)
gw.gw_ac.BSE = lib.class_as_method(BSE)
gw.gw_cd.TDA_BSE = lib.class_as_method(TDA_BSE)
gw.gw_cd.BSE = lib.class_as_method(BSE)

del (OUTPUT_THRESHOLD)
