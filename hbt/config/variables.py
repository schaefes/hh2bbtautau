# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    config.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    config.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    config.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    config.add_variable(
        name="n_hhbtag",
        expression="n_hhbtag",
        binning=(4, -0.5, 3.5),
        x_title="Number of HH b-tags",
        discrete_x=True,
    )

    # Jet Plots
    # jet 1
    config.add_variable(
        name="jet1_energy",
        expression="CollJet.E[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 1 energy",
    )
    config.add_variable(
        name="jet1_mass",
        expression="CollJet.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 1 mass",
    )
    config.add_variable(
        name="jet1_pt",
        expression="CollJet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 1 $p_{T}$",
    )
    config.add_variable(
        name="jet1_eta",
        expression="CollJet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 1 $\eta$",
    )
    config.add_variable(
        name="jet1_phi",
        expression="CollJet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 1 $\phi$",
    )
    config.add_variable(
        name="jet1_btag",
        expression="CollJet.btagDeepFlavB[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 1 b-tag",
    )

    # Jet 2
    config.add_variable(
        name="jet2_energy",
        expression="CollJet.E[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 2 energy",
    )
    config.add_variable(
        name="jet2_mass",
        expression="CollJet.mass[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 2 mass",
    )
    config.add_variable(
        name="jet2_pt",
        expression="CollJet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 2 $p_{T}$",
    )
    config.add_variable(
        name="jet2_eta",
        expression="CollJet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 2 $\eta$",
    )
    config.add_variable(
        name="jet2_phi",
        expression="CollJet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 2 $\phi$",
    )
    config.add_variable(
        name="jet2_btag",
        expression="Colljets_btag[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 2 b-tag",
    )
    # Jet 3
    config.add_variable(
        name="jet3_energy",
        expression="CollJet.E[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 3 energy",
    )
    config.add_variable(
        name="jet3_mass",
        expression="CollJet.mass[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 3 mass",
    )
    config.add_variable(
        name="jet3_pt",
        expression="CollJet.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 3 $p_{T}$",
    )
    config.add_variable(
        name="jet3_eta",
        expression="CollJet.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 3 $\eta$",
    )
    config.add_variable(
        name="jet3_phi",
        expression="CollJet.phi[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 3 $\phi$",
    )
    config.add_variable(
        name="jet3_btag",
        expression="Colljets_btag[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 3 b-tag",
    )
    # Jet 4
    config.add_variable(
        name="jet4_energy",
        expression="CollJet.E[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 4 energy",
    )
    config.add_variable(
        name="jet4_mass",
        expression="CollJet.mass[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 4 mass",
    )
    config.add_variable(
        name="jet4_pt",
        expression="CollJet.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 4 $p_{T}$",
    )
    config.add_variable(
        name="jet4_eta",
        expression="CollJet.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 4 $\eta$",
    )
    config.add_variable(
        name="jet4_phi",
        expression="CollJet.phi[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 4 $\phi$",
    )
    config.add_variable(
        name="jet4_btag",
        expression="Colljets_btag[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 4 b-tag",
    )
    # Jet 5
    config.add_variable(
        name="jet5_energy",
        expression="CollJet.E[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 5 energy",
    )
    config.add_variable(
        name="jet5_mass",
        expression="CollJet.mass[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 5 mass",
    )
    config.add_variable(
        name="jet5_pt",
        expression="CollJet.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 5 $p_{T}$",
    )
    config.add_variable(
        name="jet5_eta",
        expression="CollJet.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 5 $\eta$",
    )
    config.add_variable(
        name="jet5_phi",
        expression="CollJet.phi[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 5 $\phi$",
    )
    config.add_variable(
        name="jet5_btag",
        expression="Colljets_btag[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 5 b-tag",
    )
    # Jet 6
    config.add_variable(
        name="jet6_energy",
        expression="CollJet.E[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 6 energy",
    )
    config.add_variable(
        name="jet6_mass",
        expression="CollJet.mass[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 6 mass",
    )
    config.add_variable(
        name="jet6_pt",
        expression="CollJet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 6 $p_{T}$",
    )
    config.add_variable(
        name="jet6_eta",
        expression="CollJet.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 6 $\eta$",
    )
    config.add_variable(
        name="jet6_phi",
        expression="CollJet.phi[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 6 $\phi$",
    )
    config.add_variable(
        name="jet6_btag",
        expression="Colljets_btag[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 6 b-tag",
    )

    # VBF Jets
    config.add_variable(
        name="VBFjet1_pt",
        expression="VBFjets_pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"VBF Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="VBFjet1_mass",
        expression="VBFjets_mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 300.0),
        unit="GeV",
        x_title=r"VBF Jet 1 mass",
    )
    config.add_variable(
        name="VBFjet1_eta",
        expression="VBFjets_eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.1, 3.1),
        unit="GeV",
        x_title=r"VBF Jet 1 $\eta$",
    )
    config.add_variable(
        name="VBFjet1_phi",
        expression="VBFjets_phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"VBF Jet 1 $\phi$",
    )
    config.add_variable(
        name="VBFjet2_pt",
        expression="VBFjets_pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"VBF Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="VBFjet2_mass",
        expression="VBFjets_mass[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 300.0),
        unit="GeV",
        x_title=r"VBF Jet 2 mass",
    )
    config.add_variable(
        name="VBFjet2_eta",
        expression="VBFjets_eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.1, 3.1),
        unit="GeV",
        x_title=r"VBF Jet 2 $\eta$",
    )
    config.add_variable(
        name="VBFjet2_phi",
        expression="VBFjets_phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"VBF Jet 2 $\phi$",
    )
    config.add_variable(
        name="VBFJetsdEta",
        expression="VBFJetsdEta",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 8.0),
        unit="GeV",
        x_title=r"VBF Jets $\Delta \eta$",
    )
    config.add_variable(
        name="VBFJetsdR",
        expression="VBFJetsdR",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 8.0),
        unit="GeV",
        x_title=r"VBF Jets $\Delta$R",
    )
    config.add_variable(
        name="nVBFJets",
        expression="nVBFJets",
        null_value=EMPTY_FLOAT,
        binning=(3, 0.0, 2.0),
        unit="GeV",
        x_title=r"Number of VBF Jets",
    )

    # Tau Plots
    config.add_variable(
        name="tau1_mass",
        expression="Tau.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 5.0),
        unit="GeV",
        x_title=r"Tau 1 mass",
    )
    config.add_variable(
        name="tau1_pt",
        expression="Tau.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Tau 1 $p_{T}$",
    )
    config.add_variable(
        name="tau1_eta",
        expression="Tau.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Tau 1 $\eta$",
    )
    config.add_variable(
        name="tau1_phi",
        expression="Tau.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Tau 1 $\phi$",
    )
    config.add_variable(
        name="tau2_mass",
        expression="Tau.mass[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 5.0),
        unit="GeV",
        x_title=r"Tau 2 mass",
    )
    config.add_variable(
        name="tau2_pt",
        expression="Tau.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Tau 1 $p_{T}$",
    )
    config.add_variable(
        name="tau2_eta",
        expression="Tau.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Tau 2 $\eta$",
    )
    config.add_variable(
        name="tau2_phi",
        expression="Tau.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Tau 2 $\phi$",
    )
    config.add_variable(
        name="taudR",
        expression="TaudR",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 5.0),
        x_title=r"$\tau$ $\Delta R$",
    )
    config.add_variable(
        name="taudEta",
        expression="TaudEta",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 5.0),
        x_title=r"$\tau$ $\Delta \eta$",
    )

    # Invariant mass Plots
    config.add_variable(
        name="bjet_pair_mass",
        expression="mbjetbjet",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet Pair Mass",
    )
    config.add_variable(
        name="HH_pair_mass",
        expression="mHH",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"HH Pair Mass",
    )
    config.add_variable(
        name="tau_pair_mass",
        expression="mtautau",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Tau Pair Mass",
    )
    config.add_variable(
        name="inv_mass_d_eta",
        expression="jets_d_eta_inv_mass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 5000.0),
        unit="GeV",
        x_title=r"Invariant Mass of Jets with maximum $\Delta \eta$",
    )
    config.add_variable(
        name="hardest_jet_pair_mass",
        expression="mjj",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 800.0),
        unit="GeV",
        x_title=r"Hardest Jet Pair Mass",
    )
    config.add_variable(
        name="VBF_pair_mass",
        expression="VBFjetsInvMass",
        null_value=EMPTY_FLOAT,
        binning=(100, 0.0, 5100.0),
        unit="GeV",
        x_title=r"VBF pair Invariant Mass",
    )

    # Gen B, Tau and H Parton
    config.add_variable(
        name="GenTauInvMass",
        expression="GenPartTaupartonInvMass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Gen $\tau$ Parton $m_{\tau \tau}$",
    )
    config.add_variable(
        name="GenTaudR",
        expression="GenPartTaupartondR",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 5.0),
        unit="GeV",
        x_title=r"Gen Tau Parton $\Delta$R",
    )
    config.add_variable(
        name="GenTaudEta",
        expression="GenPartTaupartondEta",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 5.0),
        unit="GeV",
        x_title=r"Gen $\tau$ Parton $\Delta \eta$",
    )
    config.add_variable(
        name="GenTau1Mass",
        expression="GenPartTauparton1Mass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 5),
        unit="GeV",
        x_title=r"Gen $\tau_{1}$ Parton Mass",
    )
    config.add_variable(
        name="GenTau1Pt",
        expression="GenPartTauparton1Pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 600.0),
        unit="GeV",
        x_title=r"Gen $\tau_{1}$ Parton Mass",
    )
    config.add_variable(
        name="GenTau1Eta",
        expression="GenPartTauparton1Eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $\tau_{1}$ Parton $\eta$",
    )
    config.add_variable(
        name="GenTau1Phi",
        expression="GenPartTauparton1Phi",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $\tau_{1}$ Parton $\Phi$",
    )
    config.add_variable(
        name="GenTau2Mass",
        expression="GenPartTauparton2Mass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 5),
        unit="GeV",
        x_title=r"Gen $\tau_{2}$ Parton Mass",
    )
    config.add_variable(
        name="GenTau2Pt",
        expression="GenPartTauparton2Pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 600.0),
        unit="GeV",
        x_title=r"Gen $\tau_{2}$ Parton Mass",
    )
    config.add_variable(
        name="GenTau2Eta",
        expression="GenPartTauparton2Eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $\tau_{2}$ Parton $\eta$",
    )
    config.add_variable(
        name="GenTau2Phi",
        expression="GenPartTauparton2Phi",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $\tau_{2}$ Parton $\Phi$",
    )

    config.add_variable(
        name="GenBInvMass",
        expression="GenPartBpartonInvMass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Gen B Parton $m_{bb}$",
    )
    config.add_variable(
        name="GenBdR",
        expression="GenPartBpartondR",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 5.0),
        unit="GeV",
        x_title=r"Gen B Parton $\Delta$R",
    )
    config.add_variable(
        name="GenBdR_1",
        expression="GenPartBpartondR_1",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 5.0),
        unit="GeV",
        x_title=r"Gen B Parton $\Delta$R_1",
    )
    config.add_variable(
        name="GenBdEta",
        expression="GenPartBpartondEta",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 5.0),
        unit="GeV",
        x_title=r"Gen B Parton $\Delta \eta$",
    )
    config.add_variable(
        name="GenB1Mass",
        expression="GenPartBparton1Mass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Gen $B_{1}$ Parton Mass",
    )
    config.add_variable(
        name="GenB1Pt",
        expression="GenPartBparton1Pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Gen $B_{1}$ Parton Mass",
    )
    config.add_variable(
        name="GenB1Eta",
        expression="GenPartBparton1Eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $B_{1}$ Parton $\eta$",
    )
    config.add_variable(
        name="GenB1Phi",
        expression="GenPartBparton1Phi",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $B_{1}$ Parton $\Phi$",
    )
    config.add_variable(
        name="GenB2Mass",
        expression="GenPartBparton2Mass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Gen $B_{2}$ Parton Mass",
    )
    config.add_variable(
        name="GenB2Pt",
        expression="GenPartBparton2Pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Gen $B_{2}$ Parton Mass",
    )
    config.add_variable(
        name="GenB2Eta",
        expression="GenPartBparton2Eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $B_{2}$ Parton $\eta$",
    )
    config.add_variable(
        name="GenB2Phi",
        expression="GenPartBparton2Phi",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $B_{2}$ Parton $\Phi$",
    )

    config.add_variable(
        name="GenHInvMass",
        expression="GenPartHpartonInvMass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 1300.0),
        unit="GeV",
        x_title=r"Gen H Parton $m_{HH}$",
    )
    config.add_variable(
        name="GenHdR",
        expression="GenPartHpartondR",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 5.0),
        unit="GeV",
        x_title=r"Gen H Parton $\Delta$R",
    )
    config.add_variable(
        name="GenHdEta",
        expression="GenPartHpartondEta",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 5.0),
        unit="GeV",
        x_title=r"Gen H Parton $\Delta \eta$",
    )
    config.add_variable(
        name="GenH1Mass",
        expression="GenPartHparton1Mass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 200),
        unit="GeV",
        x_title=r"Gen $H_{1}$ Parton Mass",
    )
    config.add_variable(
        name="GenH1Pt",
        expression="GenPartHparton1Pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 600.0),
        unit="GeV",
        x_title=r"Gen $H_{1}$ Parton $p_{T}$",
    )
    config.add_variable(
        name="GenH1Eta",
        expression="GenPartHparton1Eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $H_{1}$ Parton $\eta$",
    )
    config.add_variable(
        name="GenH1Phi",
        expression="GenPartHparton1Phi",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $H_{1}$ Parton $\Phi$",
    )
    config.add_variable(
        name="GenH1Gamma",
        expression="GenHparton1Gamma",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 30.0),
        unit="GeV",
        x_title=r"Gen $H_{1}$ Lorentz Factor $\gamma$",
    )
    config.add_variable(
        name="GenH2Mass",
        expression="GenPartHparton2Mass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 200),
        unit="GeV",
        x_title=r"Gen $H_{2}$ Parton Mass",
    )
    config.add_variable(
        name="GenH2Pt",
        expression="GenPartHparton2Pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 600.0),
        unit="GeV",
        x_title=r"Gen $H_{2}$ Parton $p_{T}$",
    )
    config.add_variable(
        name="GenH2Eta",
        expression="GenPartHparton2Eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $H_{2}$ Parton $\eta$",
    )
    config.add_variable(
        name="GenH2Phi",
        expression="GenPartHparton2Phi",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen $H_{2}$ Parton $\Phi$",
    )
    config.add_variable(
        name="GenH2Gamma",
        expression="GenHparton2Gamma",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 30.0),
        unit="GeV",
        x_title=r"Gen $H_{2}$ Lorentz Factor $\gamma$",
    )
    config.add_variable(
        name="GenHHGamma",
        expression="GenHHGamma",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 30.0),
        unit="GeV",
        x_title=r"Gen HH Lorentz Factor $\gamma$",
    )

    config.add_variable(
        name="GenVBFInvMass",
        expression="GenPartVBFpartonInvMass",
        null_value=EMPTY_FLOAT,
        binning=(100, 0.0, 5100.0),
        unit="GeV",
        x_title=r"Gen VBF Parton $m_{VBF1,VBF2}$",
    )
    config.add_variable(
        name="GenVBFdR",
        expression="GenPartVBFpartondR",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 8.0),
        unit="GeV",
        x_title=r"Gen VBF Parton $\Delta$R",
    )
    config.add_variable(
        name="GenVBFdEta",
        expression="GenPartVBFpartondEta",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 8.0),
        unit="GeV",
        x_title=r"Gen VBF Parton $\Delta \eta$",
    )
    config.add_variable(
        name="GenVBF1Mass",
        expression="GenPartVBFparton1Mass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Gen VBF1 Parton Mass",
    )
    config.add_variable(
        name="GenVBF1Pt",
        expression="GenPartVBFparton1Pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 600.0),
        unit="GeV",
        x_title=r"Gen VBF1 Parton $p_{T}$",
    )
    config.add_variable(
        name="GenVBF1Eta",
        expression="GenPartVBFparton1Eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.1, 3.1),
        unit="GeV",
        x_title=r"Gen VBF1 Parton $\eta$",
    )
    config.add_variable(
        name="GenVBF1Phi",
        expression="GenPartVBFparton1Phi",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen VBF1 Parton $\Phi$",
    )
    config.add_variable(
        name="GenVBF1Gamma",
        expression="GenVBFparton1Gamma",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 30.0),
        unit="GeV",
        x_title=r"Gen VBF1 Lorentz Factor $\gamma$",
    )
    config.add_variable(
        name="GenVBF2Mass",
        expression="GenPartVBFparton2Mass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Gen VBF2 Parton Mass",
    )
    config.add_variable(
        name="GenVBF2Pt",
        expression="GenPartVBFparton2Pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 600.0),
        unit="GeV",
        x_title=r"Gen VBF2 Parton $p_{T}$",
    )
    config.add_variable(
        name="GenVBF2Eta",
        expression="GenPartVBFparton2Eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.1, 3.1),
        unit="GeV",
        x_title=r"Gen VBF2 Parton $\eta$",
    )
    config.add_variable(
        name="GenVBF2Phi",
        expression="GenPartVBFparton2Phi",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Gen VBF2 Parton $\Phi$",
    )
    config.add_variable(
        name="GenVBF2Gamma",
        expression="GenVBFparton2Gamma",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 30.0),
        unit="GeV",
        x_title=r"Gen VBF2 Lorentz Factor $\gamma$",
    )

    # Gen Matched VBF Jets
    config.add_variable(
        name="GenMatchedVBFJet1Pt",
        expression="GenMatchedVBFJets.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"Gen Matched VBF Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="GenMatchedVBFJet2Pt",
        expression="GenMatchedVBFJets.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"Gen Matched VBF Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="GenMatchedVBFJet1Phi",
        expression="GenMatchedVBFJets.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched VBF Jet 1 $\phi$",
    )
    config.add_variable(
        name="GenMatchedVBFJet2Phi",
        expression="GenMatchedVBFJets.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched VBF Jet 2 $\phi$",
    )
    config.add_variable(
        name="GenMatchedVBFJet1Eta",
        expression="GenMatchedVBFJets.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(36, -4.8, 4.8),
        unit="GeV",
        x_title=r"Gen Matched VBF Jet 1 $\eta$",
    )
    config.add_variable(
        name="GenMatchedVBFJet2Eta",
        expression="GenMatchedVBFJets.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched VBF Jet 2 $\eta$",
    )
    # Automatic VBF GenJet to Jet matching
    config.add_variable(
        name="AutoGenMatchedVBFJet1Pt",
        expression="AutoGenMatchedVBFJets.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"Auto Gen Matched VBF Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="AutoGenMatchedVBFJet2Pt",
        expression="AutoGenMatchedVBFJets.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"Auto Gen Matched VBF Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="AutoGenMatchedVBFJet2Eta",
        expression="AutoGenMatchedVBFJets.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Auto Gen Matched VBF Jet 2 $\eta$",
    )
    config.add_variable(
        name="AutoGenMatchedVBFJet1Eta",
        expression="AutoGenMatchedVBFJets.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Auto Gen Matched VBF Jet 1 $\eta$",
    )
    config.add_variable(
        name="AutoGenMatchedVBFJet2Phi",
        expression="AutoGenMatchedVBFJets.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Auto Gen Matched VBF Jet 2 $\phi$",
    )
    config.add_variable(
        name="AutoGenMatchedVBFJet1Phi",
        expression="AutoGenMatchedVBFJets.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Auto Gen Matched VBF Jet 1 $\phi$",
    )

    # Gen Matched GenVBF Jets
    config.add_variable(
        name="GenMatchedGenVBFJet1Pt",
        expression="genMatchedGenVBFJets.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"Gen Matched Gen VBF Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="GenMatchedGenVBFJet2Pt",
        expression="genMatchedGenVBFJets.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"Gen Matched Gen VBF Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="GenMatchedGenVBFJet1Phi",
        expression="genMatchedGenVBFJets.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched Gen VBF Jet 1 $\phi$",
    )
    config.add_variable(
        name="GenMatchedGenVBFJet2Phi",
        expression="genMatchedGenVBFJets.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched Gen VBF Jet 2 $\phi$",
    )
    config.add_variable(
        name="GenMatchedGenVBFJet1Eta",
        expression="genMatchedGenVBFJets.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(36, -4.8, 4.8),
        unit="GeV",
        x_title=r"Gen Matched Gen VBF Jet 1 $\eta$",
    )
    config.add_variable(
        name="GenMatchedGenVBFJet2Eta",
        expression="genMatchedGenVBFJets.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched Gen VBF Jet 2 $\eta$",
    )

    # Gen Matched B Jets
    config.add_variable(
        name="GenMatchedBJet1Pt",
        expression="GenMatchedBJets.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"Gen Matched BJet 1 $p_{T}$",
    )
    config.add_variable(
        name="GenMatchedBJet2Pt",
        expression="GenMatchedBJets.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"Gen Matched BJet 2 $p_{T}$",
    )
    config.add_variable(
        name="GenMatchedBJet1Phi",
        expression="GenMatchedBJets.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched BJet 1 $\phi$",
    )
    config.add_variable(
        name="GenMatchedBJet2Phi",
        expression="GenMatchedBJets.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched BJet 2 $\phi$",
    )
    config.add_variable(
        name="GenMatchedBJet1Eta",
        expression="GenMatchedBJets.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(36, -4.8, 4.8),
        unit="GeV",
        x_title=r"Gen Matched BJet 1 $\eta$",
    )
    config.add_variable(
        name="GenMatchedBJet2Eta",
        expression="GenMatchedBJets.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched BJet 2 $\eta$",
    )

    # Gen Matched B Jets
    config.add_variable(
        name="GenMatchedGenBJet1Pt",
        expression="genMatchedGenBJets.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"Gen Matched Gen BJet 1 $p_{T}$",
    )
    config.add_variable(
        name="GenMatchedGenBJet2Pt",
        expression="genMatchedGenBJets.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"Gen Matched Gen BJet 2 $p_{T}$",
    )
    config.add_variable(
        name="GenMatchedGenBJet1Phi",
        expression="genMatchedGenBJets.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched Gen BJet 1 $\phi$",
    )
    config.add_variable(
        name="GenMatchedGenBJet2Phi",
        expression="genMatchedGenBJets.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched Gen BJet 2 $\phi$",
    )
    config.add_variable(
        name="GenMatchedGenBJet1Eta",
        expression="genMatchedGenBJets.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(36, -4.8, 4.8),
        unit="GeV",
        x_title=r"Gen Matched Gen BJet 1 $\eta$",
    )
    config.add_variable(
        name="GenMatchedGenBJet2Eta",
        expression="genMatchedGenBJets.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        unit="GeV",
        x_title=r"Gen Matched Gen BJet 2 $\eta$",
    )
    # Others
    config.add_variable(
        name="max_d_eta",
        expression="jets_max_d_eta",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 8.0),
        x_title=r"Maximum $\Delta \eta$ of Jets",
    )
    config.add_variable(
        name="hardest_jet_pair_pt",
        expression="hardest_jet_pair_pt",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 900.0),
        unit="GeV",
        x_title=r"$p_{T}$ of hardest Jets",
    )
    config.add_variable(
        name="ht",
        binning=(50, 0.0, 1200.0),
        unit="GeV",
        x_title="HT",
    )
    config.add_variable(
        name="n_jet",
        expression="n_jets",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    config.add_variable(
        name="energy_corr",
        expression="energy_corr",
        binning=(100, 700, 2000000),
        unit=r"$GeV^{2}$",
        x_title="Energy Correlation",
    )
    config.add_variable(
        name="met_phi",
        expression="MET.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )
    config.add_variable(
        name="Btagging_efficiency",
        expression="Btagging_results",
        null_value=EMPTY_FLOAT,
        binning=(3, 0.0, 2.01),
        unit="GeV",
        x_title=r"B tagging efficiency",
    )
    config.add_variable(
        name="HHBtagging_efficiency",
        expression="HHBtagging_results",
        null_value=EMPTY_FLOAT,
        binning=(3, 0.0, 2.01),
        unit="GeV",
        x_title=r"HHB tagging efficiency",
    )
    config.add_variable(
        name="VBFtagging_efficiency_auto",
        expression="VBFtagging_results_auto",
        null_value=EMPTY_FLOAT,
        binning=(4, -1.0, 2.01),
        unit="GeV",
        x_title=r"VBF Jets tagging efficiency (Automatic Matching)",
    )
    config.add_variable(
        name="VBFtagging_efficiency_dr",
        expression="VBFtagging_results_dr",
        null_value=EMPTY_FLOAT,
        binning=(4, -1.0, 2.01),
        unit="GeV",
        x_title=r"VBF Jets tagging efficiency ($\Delta$R Matching)",
    )

    # B Jets
    config.add_variable(
        name="bjet1_mass",
        expression="BJet.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet 1 mass",
    )
    config.add_variable(
        name="bjet1_pt",
        expression="BJet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$",
    )
    config.add_variable(
        name="bjet1_eta",
        expression="BJet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"BJet 1 $\eta$",
    )
    config.add_variable(
        name="bjet1_phi",
        expression="BJet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"BJet 1 $\phi$",
    )
    config.add_variable(
        name="bjet2_mass",
        expression="BJet.mass[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet 2 mass",
    )
    config.add_variable(
        name="bjet2_pt",
        expression="BJet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$",
    )
    config.add_variable(
        name="bjet2_eta",
        expression="BJet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"BJet 2 $\eta$",
    )
    config.add_variable(
        name="bjet2_phi",
        expression="BJet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"BJet 2 $\phi$",
    )

    config.add_variable(
        name="hhbjet1_eta",
        expression="HHBJet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"HHBJet 1 $\eta$",
    )
    config.add_variable(
        name="BJetsdR",
        expression="BJetsdR",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 7.0),
        x_title=r"BJets $\Delta R$",
    )
    config.add_variable(
        name="BJetsEta",
        expression="BJetsdEta",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 7.0),
        x_title=r"BJets $\Delta \eta$",
    )

    # weights
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    config.add_variable(
        name="pu_weight",
        expression="pu_weight",
        binning=(40, 0, 2),
        x_title="Pileup weight",
    )
    config.add_variable(
        name="normalized_pu_weight",
        expression="normalized_pu_weight",
        binning=(40, 0, 2),
        x_title="Normalized pileup weight",
    )
    config.add_variable(
        name="btag_weight",
        expression="btag_weight",
        binning=(60, 0, 3),
        x_title="b-tag weight",
    )
    config.add_variable(
        name="normalized_btag_weight",
        expression="normalized_btag_weight",
        binning=(60, 0, 3),
        x_title="Normalized b-tag weight",
    )
    config.add_variable(
        name="normalized_njet_btag_weight",
        expression="normalized_njet_btag_weight",
        binning=(60, 0, 3),
        x_title="$N_{jet}$ normalized b-tag weight",
    )

    # cutflow variables
    config.add_variable(
        name="cf_njet",
        expression="cutflow.n_jet",
        binning=(17, -0.5, 16.5),
        x_title="Jet multiplicity",
        discrete_x=True,
    )
    config.add_variable(
        name="cf_ht",
        expression="cutflow.ht",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$H_{T}$",
    )
    config.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet1_eta",
        expression="cutflow.jet1_eta",
        binning=(40, -5.0, 5.0),
        x_title=r"Jet 1 $\eta$",
    )
    config.add_variable(
        name="cf_jet1_phi",
        expression="cutflow.jet1_phi",
        binning=(32, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$",
    )
    config.add_variable(
        name="cf_jet2_pt",
        expression="cutflow.jet2_pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet2_eta",
        expression="cutflow.jet1_eta",
        null_value=EMPTY_FLOAT,
        binning=(40, -5.0, 5.0),
        x_title=r"Jet 2 $\eta$",
    )
    config.add_variable(
        name="cf_jet3_pt",
        expression="cutflow.jet3_pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 3 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet4_pt",
        expression="cutflow.jet4_pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 4 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet5_pt",
        expression="cutflow.jet5_pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 5 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet6_pt",
        expression="cutflow.jet6_pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )

