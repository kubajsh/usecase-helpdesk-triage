
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Hardcoded Knowledge Base
GRC_KNOWLEDGE_BASE = [
    {"Ticket ID": "INC1254888", "Description": "A user requests to be exempt from a security policy (e.g., 'Need to use a non-standard app').", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254889", "Description": "The task for a manager or GRC analyst to formally approve or deny an exception request.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_POLICY_OWNERS_24"},
    {"Ticket ID": "INC1254890", "Description": "An annual or ad-hoc task to review and update the 'Password Policy.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254891", "Description": "A workflow to get a new 'Cloud Security Policy' approved by legal, security, and the CIO.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254892", "Description": "A task to map the requirements of a new law (e.g., NIS2) to existing security controls.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254893", "Description": "A task sent to a system owner to certify, 'Yes, I review access logs for this server monthly.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254894", "Description": "An issue created when a control owner attests 'No,' they are not compliant.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254895", "Description": "An issue generated from a test, e.g., 'Firewall rule review control failed its test.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_IT_INFRA_24"},
    {"Ticket ID": "INC1254896", "Description": "A task assigned to an admin to 'Upload evidence of firewall rule reviews for Q3.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254897", "Description": "A task to assess a new business unit against the company's ISO 27001 controls.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_SECURITY_OPS_24"},
    {"Ticket ID": "INC1254898", "Description": "An issue raised from a DLP system, e.g., 'User attempted to email PII to a personal address.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_HR_MGMT_24"},
    {"Ticket ID": "INC1254899", "Description": "An issue for a user accessing prohibited websites on a corporate device.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_DATA_OWNERS_24"},
    {"Ticket ID": "INC1254900", "Description": "An issue where a database is marked 'Public' but contains 'Confidential' data.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_IT_INFRA_24"},
    {"Ticket ID": "INC1254901", "Description": "A specific task to fix a failed control, e.g., 'Implement MFA on system X.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_HR_MGMT_24"},
    {"Ticket ID": "INC1254902", "Description": "An issue tracking employees who are overdue for mandatory security awareness training.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254903", "Description": "A task to define a new control for a new technology (e.g., 'Secure use of generative AI').", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254904", "Description": "A task to review a new framework version (e.g., NIST 800-53 Rev. 5) and update controls.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254905", "Description": "A workflow to retire a control that is no longer applicable (e.g., for a legacy system).", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_LEGAL_PRIVACY_24"},
    {"Ticket ID": "INC1254906", "Description": "A high-level record to manage a significant compliance failure or investigation.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254907", "Description": "A task to initiate the quarterly self-assessment for PCI-DSS controls.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254908", "Description": "A record for a newly identified risk, e.g., 'Risk of ransomware attack on critical servers.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254909", "Description": "A task assigned to a risk owner to determine the likelihood and impact of a risk.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254910", "Description": "A task to create a plan to 'Mitigate,' 'Avoid,' 'Transfer,' or 'Accept' a high-level risk.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_IT_INFRA_24"},
    {"Ticket ID": "INC1254911", "Description": "A specific action to reduce a risk, e.g., 'Install EDR solution on all endpoints.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_LEADERSHIP_24"},
    {"Ticket ID": "INC1254912", "Description": "A formal request from a business owner to 'Accept' a risk (e.g., 'Accept risk of not patching').", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_LEADERSHIP_24"},
    {"Ticket ID": "INC1254913", "Description": "A task for senior management (e.g., CISO) to approve the acceptance of a known risk.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254914", "Description": "A periodic task to review and re-assess all risks in the IT risk register.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254915", "Description": "An issue created when a Key Risk Indicator is breached, e.g., 'Number of high-sev vulnerabilities > 500.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254916", "Description": "A task to gather and report KRI data.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_SECURITY_OPS_24"},
    {"Ticket ID": "INC1254917", "Description": "A record of a risk that has occurred, e.g., 'User credentials compromised and used by an attacker.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254918", "Description": "A task to investigate why a risk event occurred and how to prevent it.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_PMO_24"},
    {"Ticket ID": "INC1254919", "Description": "A mandatory task for any new IT project to identify its security risks.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_THREAT_INTEL_24"},
    {"Ticket ID": "INC1254920", "Description": "A task to analyze a new threat (e.g., 'Log4j') against the organization's assets.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254921", "Description": "A record linking a specific risk (e.g., 'Data exfiltration') to a specific asset (e.g., 'Customer DB').", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254922", "Description": "A task to re-evaluate a risk's score after mitigation controls have been applied.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254923", "Description": "A task to roll up multiple low-level risks (e.g., '20 unpatched servers') into one high-level risk.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_LEGAL_PRIVACY_24"},
    {"Ticket ID": "INC1254924", "Description": "A task to assess the privacy risk of a new system collecting PII.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_CLOUD_OPS_24"},
    {"Ticket ID": "INC1254925", "Description": "A task to evaluate the risks of migrating a new application to AWS/Azure/GCP.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_AI_GOVERNANCE_24"},
    {"Ticket ID": "INC1254926", "Description": "A task to identify risks (bias, data poisoning, privacy) in a new generative AI model.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254927", "Description": "A request to delay the fix for a risk, with justification.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254928", "Description": "A task to schedule and scope an upcoming internal audit (e.g., 'Q4 SOX ITGC Audit').", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254929", "Description": "A task assigned to a control owner to 'Provide a list of all users with admin access to SAP.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254930", "Description": "The record of the evidence provided by the control owner, awaiting auditor review.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254931", "Description": "A task for the auditor to review the submitted evidence for sufficiency and effectiveness.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254932", "Description": "An issue created by an auditor, e.g., 'Finding: Segregation of Duties violation found in SAP.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254933", "Description": "A task for the finding owner to create a plan to fix the issue.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254934", "Description": "A record of the business owner's formal response ('Agree' or 'Disagree') to the audit finding.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_IT_INFRA_24"},
    {"Ticket ID": "INC1254935", "Description": "The specific task to fix the finding (e.g., 'Remove conflicting roles from 5 user accounts').", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254936", "Description": "A task for the auditor to verify that the remediation was effective.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254937", "Description": "A task to document the steps an auditor will take to test a control.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254938", "Description": "A scheduled task to interview a control owner and 'walk through' the control process.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254939", "Description": "A 'soft' finding or recommendation that is not a direct failure but an area for improvement.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254940", "Description": "A task to compile all findings, observations, and outcomes into a final report.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254941", "Description": "A task for an audit manager to assign auditors to an engagement.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254942", "Description": "A task to manage evidence requests from an external auditor (e.g., PwC, Deloitte).", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254943", "Description": "A record defining the test steps for a specific security control (e.g., '1. Pull user list...').", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254944", "Description": "A scheduled interview with a key process owner.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254945", "Description": "An issue that is escalated to senior management due to high risk or non-remediation.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_INTERNAL_AUDIT_24"},
    {"Ticket ID": "INC1254946", "Description": "A task to define which applications, locations, and controls are in-scope for an audit.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254947", "Description": "An alert from an automated test, e.g., 'Automated script detected a privileged account without MFA.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VENDOR_MGMT_24"},
    {"Ticket ID": "INC1254948", "Description": "A workflow to initiate a security review for a new potential vendor.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VENDOR_MGMT_24"},
    {"Ticket ID": "INC1254949", "Description": "A task to classify a vendor as 'Critical,' 'High,' 'Medium,' or 'Low' risk.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "VENDOR_EXTERNAL_CONTACTS_24"},
    {"Ticket ID": "INC1254950", "Description": "The task of sending a security questionnaire (e.g., SIG, CAIQ) to a vendor.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VENDOR_MGMT_24"},
    {"Ticket ID": "INC1254951", "Description": "A task for a GRC analyst to review the vendor's questionnaire responses.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VENDOR_MGMT_24"},
    {"Ticket ID": "INC1254952", "Description": "An issue created from a vendor's assessment, e.g., 'Vendor does not have a formal BCP plan.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "VENDOR_EXTERNAL_CONTACTS_24"},
    {"Ticket ID": "INC1254953", "Description": "A task to track the vendor's plan to fix an identified finding.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_LEGAL_PRIVACY_24"},
    {"Ticket ID": "INC1254954", "Description": "A task for legal and security to review the security clauses in a vendor's contract.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_IT_INFRA_24"},
    {"Ticket ID": "INC1254955", "Description": "A workflow to ensure vendor access is revoked and data is returned/destroyed.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VENDOR_MGMT_24"},
    {"Ticket ID": "INC1254956", "Description": "An annual task to re-evaluate the risk of a critical vendor.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VENDOR_MGMT_24"},
    {"Ticket ID": "INC1254957", "Description": "An issue to manage a security breach at a vendor that affects your company.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VENDOR_MGMT_24"},
    {"Ticket ID": "INC1254958", "Description": "A task to obtain and review a vendor's SOC 2 Type II report for exceptions.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VENDOR_MGMT_24"},
    {"Ticket ID": "INC1254959", "Description": "An issue to track a critical subcontractor used by your vendor (a 4th party).", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254960", "Description": "A periodic task to review a vendor's security performance (e.g., SLA compliance).", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_LEGAL_PRIVACY_24"},
    {"Ticket ID": "INC1254961", "Description": "A task to ensure a Data Processing Agreement (DPA) is in place for any vendor handling PII.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VENDOR_MGMT_24"},
    {"Ticket ID": "INC1254962", "Description": "Escalating a vendor's non-compliance to the business relationship owner.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254963", "Description": "An issue to track an outage at a key vendor and activate BCP plans.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VENDOR_MGMT_24"},
    {"Ticket ID": "INC1254964", "Description": "A task to plan and conduct a physical or remote audit of a high-risk vendor.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VENDOR_MGMT_24"},
    {"Ticket ID": "INC1254965", "Description": "A task to ensure a vendor has appropriate cyber insurance.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_LEADERSHIP_24"},
    {"Ticket ID": "INC1254966", "Description": "A task to request and review the executive summary of a vendor's latest pen test.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_IT_INFRA_24"},
    {"Ticket ID": "INC1254967", "Description": "A formal acceptance record for a risk identified with a vendor that cannot be mitigated.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_IT_INFRA_24"},
    {"Ticket ID": "INC1254968", "Description": "A task assigned to a server admin to 'Patch CVE-2025-12345.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254969", "Description": "A request to delay patching a vulnerability (a type of policy exception).", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_LEADERSHIP_24"},
    {"Ticket ID": "INC1254970", "Description": "The task for a manager or CISO to approve the deferral.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VULN_MGMT_24"},
    {"Ticket ID": "INC1254971", "Description": "An issue created when a critical vulnerability is not patched within its 30-day SLA.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254972", "Description": "A GRC task to link a group of vulnerabilities to a high-risk business asset, raising its priority.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254973", "Description": "A security incident (e.g., 'Malware Outbreak') is escalated to GRC to be tracked as a 'Risk Event.'", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254974", "Description": "A task for GRC to analyze phishing campaign data to identify a new 'Risk' or update KRI metrics.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_IT_INFRA_24"},
    {"Ticket ID": "INC1254975", "Description": "An issue from a scanner that a server fails a CIS benchmark (e.g., 'SSH root login enabled').", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_IT_INFRA_24"},
    {"Ticket ID": "INC1254976", "Description": "An issue for a patch that was deployed but failed to install correctly.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_APP_DEV_24"},
    {"Ticket ID": "INC1254977", "Description": "An issue from a SAST/DAST scan (e.g., 'SQL Injection vulnerability in code').", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_LEGAL_PRIVACY_24"},
    {"Ticket ID": "INC1254978", "Description": "An issue from a DLP tool that is escalated to GRC for compliance and privacy review.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_SECURITY_OPS_24"},
    {"Ticket ID": "INC1254979", "Description": "A GRC 'case' to support an investigation, linking policy violations and system alerts.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_VULN_MGMT_24"},
    {"Ticket ID": "INC1254980", "Description": "A task to move a vulnerability ticket to the correct remediation team.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254981", "Description": "A task to update an asset's business criticality, which impacts risk and vulnerability priority.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254982", "Description": "A task to determine the Recovery Time/Point Objective (RTO/RPO) of an application.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_BCM_TEAM_24"},
    {"Ticket ID": "INC1254983", "Description": "A task to update the Business Continuity Plan for a critical business service.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_BCM_TEAM_24"},
    {"Ticket ID": "INC1254984", "Description": "A task to plan and execute a tabletop exercise to test the BCM plan.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_ASSET_OWNERS_24"},
    {"Ticket ID": "INC1254985", "Description": "An issue created when a BCM test fails (e.g., 'DR site failover took 8 hours; RTO is 4').", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_BCM_TEAM_24"},
    {"Ticket ID": "INC1254986", "Description": "A high-level record activated during a major event (e.g., 'Data center fire').", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"},
    {"Ticket ID": "INC1254987", "Description": "A task to review a major incident and create new risks or controls to prevent recurrence.", "assigment_group": "CUSTOMER_HELPDESK_SD24", "resolver_group": "CUSTOMER_GRC_ANALYSTS_24"}
]

app = Flask(__name__)

# 2. Pre-process the knowledge base on startup
vectorizer = TfidfVectorizer(stop_words='english')
knowledge_base_descriptions = [item["Description"] for item in GRC_KNOWLEDGE_BASE]
knowledge_base_vectors = vectorizer.fit_transform(knowledge_base_descriptions)

# HTML Template for the frontend
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GRC Incident Assigner</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f7f7f7;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background-color: #fff;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 700px;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        textarea {
            width: 100%;
            padding: 0.75rem;
            border-radius: 4px;
            border: 1px solid #ddd;
            font-size: 1rem;
            margin-bottom: 1rem;
            box-sizing: border-box;
            resize: vertical;
            min-height: 120px;
        }
        button {
            display: block;
            width: 100%;
            padding: 0.75rem;
            border: none;
            border-radius: 4px;
            background-color: #007bff;
            color: white;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: #0056b3;
        }
        #results {
            margin-top: 2rem;
            display: none;
        }
        #loading {
            text-align: center;
            display: none;
            color: #555;
        }
        .result-main {
            background-color: #e7f3ff;
            border-left: 5px solid #007bff;
            padding: 1rem;
            margin-bottom: 1.5rem;
            font-size: 1.1rem;
        }
        .result-main strong {
            font-weight: 600;
        }
        h3 {
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        .visual-path-list {
            list-style: none;
            padding: 0;
        }
        .visual-path-item {
            background-color: #f9f9f9;
            border: 1px solid #eee;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-radius: 4px;
        }
        .visual-path-item p {
            margin: 0 0 0.5rem 0;
        }
        .visual-path-item .ticket-id {
            font-weight: bold;
            color: #007bff;
        }
        .visual-path-item .score {
            font-style: italic;
            color: #28a745;
        }
        .timeline {
            list-style-type: none;
            padding-left: 20px;
            position: relative;
        }
        .timeline:before {
            content: '';
            position: absolute;
            left: 20px;
            top: 0;
            bottom: 0;
            width: 2px;
            background-color: #ddd;
        }
        .timeline-item {
            margin-bottom: 20px;
            position: relative;
            padding-left: 30px;
        }
        .timeline-marker {
            position: absolute;
            left: 12px;
            top: 5px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background-color: #007bff;
            border: 2px solid #fff;
        }
        .timeline-content {
            background-color: #f9f9f9;
            padding: 10px 15px;
            border-radius: 4px;
            border: 1px solid #eee;
        }
        .timeline-content .stage {
            font-weight: bold;
            margin: 0 0 5px 0;
        }
        .timeline-content .time {
            font-size: 0.9em;
            color: #555;
            margin: 0;
        }
        .results-grid {
            display: flex;
            gap: 2rem;
        }
        .left-column, .right-column {
            flex: 1;
        }
        .timeline {
            list-style-type: none;
            padding-left: 0;
            position: relative;
        }
        .timeline:before {
            content: ' ';
            background: #d4d9df;
            display: inline-block;
            position: absolute;
            left: 8px;
            width: 2px;
            height: 100%;
            z-index: 400;
        }
        .timeline-item {
            margin: 20px 0;
            padding-left: 30px;
        }
        .timeline-marker {
            position: absolute;
            left: 0;
            top: 0.3em;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background-color: #007bff;
            border: 3px solid #fff;
            z-index: 500;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>GRC Incident Assigner</h1>
        <form id="assign-form">
            <textarea id="description" placeholder="Enter the incident description here..." required></textarea>
            <button type="submit">Submit</button>
        </form>
        <div id="loading">Loading...</div>
        <div id="results">
            <div class="result-main">
                Assigning ticket to: <strong id="predicted-group"></strong>
            </div>
            <h3>Justification (Based on similar tickets):</h3>
            <div class="results-grid">
                <div class="left-column">
                    <div id="visual-path-container"></div>
                </div>
                <div class="right-column">
                    <h3>Past Ticket Lifecycle</h3>
                    <div id="visual-flow-container"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('assign-form').addEventListener('submit', async function(event) {
            event.preventDefault();
            
            const description = document.getElementById('description').value;
            const resultsDiv = document.getElementById('results');
            const loadingDiv = document.getElementById('loading');
            const submitButton = this.querySelector('button');

            resultsDiv.style.display = 'none';
            loadingDiv.style.display = 'block';
            submitButton.disabled = true;

            try {
                const response = await fetch('/assign', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ description: description })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                
                document.getElementById('predicted-group').textContent = data.predicted_group;
                
                const visualPathContainer = document.getElementById('visual-path-container');
                visualPathContainer.innerHTML = ''; // Clear previous results
                
                const ol = document.createElement('ol');
                ol.className = 'visual-path-list';

                data.visual_path.forEach(item => {
                    const li = document.createElement('li');
                    li.className = 'visual-path-item';
                    li.innerHTML = `
                        <p><span class="ticket-id">${item.ticket_id}</span></p>
                        <p>${item.description}</p>
                        <p><span class="score">Similarity Score: ${item.similarity_score}</span></p>
                    `;
                    ol.appendChild(li);
                });
                visualPathContainer.appendChild(ol);

                const visualFlowContainer = document.getElementById('visual-flow-container');
                visualFlowContainer.innerHTML = '';
                const flowUl = document.createElement('ul');
                flowUl.className = 'timeline';
                data.visual_flow.forEach(item => {
                    const flowLi = document.createElement('li');
                    flowLi.className = 'timeline-item';
                    flowLi.innerHTML = `
                        <div class="timeline-marker"></div>
                        <div class="timeline-content">
                            <p class="stage">${item.stage}</p>
                            <p class="group">Group: ${item.group}</p>
                            <p class="time">Time Elapsed: ${item.time_spent}</p>
                        </div>
                    `;
                    flowUl.appendChild(flowLi);
                });
                visualFlowContainer.appendChild(flowUl);

                resultsDiv.style.display = 'block';

            } catch (error) {
                console.error("Error fetching assignment:", error);
                alert("An error occurred. Please check the console for details.");
            } finally {
                loadingDiv.style.display = 'none';
                submitButton.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/assign', methods=['POST'])
def assign_incident():
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({"error": "Description not provided"}), 400

    new_description = data['description']
    
    # Transform the new description
    new_vector = vectorizer.transform([new_description])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(new_vector, knowledge_base_vectors).flatten()
    
    # Get top 3 matches
    top_3_indices = np.argsort(similarities)[-3:][::-1]
    
    # Format the response
    predicted_group = GRC_KNOWLEDGE_BASE[top_3_indices[0]]["resolver_group"]
    
    visual_path = []
    for i in top_3_indices:
        ticket = GRC_KNOWLEDGE_BASE[i]
        score = similarities[i]
        visual_path.append({
            "ticket_id": ticket["Ticket ID"],
            "description": ticket["Description"],
            "resolver_group": ticket["resolver_group"],
            "similarity_score": f"{score:.1%}"
        })
        
    response = {
        "predicted_group": predicted_group,
        "visual_path": visual_path,
        "visual_flow": [
            {
                "stage": "Ticket Created",
                "group": "CUSTOMER_HELPDESK_24",
                "time_spent": "0 hours"
            },
            {
                "stage": "Ticket Assigned",
                "group": predicted_group,
                "time_spent": "2 hours"
            },
            {
                "stage": "Ticket Resolved",
                "group": predicted_group,
                "time_spent": "8 hours"
            }
        ]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
