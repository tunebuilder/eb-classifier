
import streamlit as st
import yaml
import os
from pathlib import Path
from pdf_processor import PDFProcessor
from llm_client import LLMClient
from results_manager import ResultsManager

st.set_page_config(page_title="Evidence Base Classifier")

# --- Config file management ---
def get_config_path():
    """Get the path to the config file."""
    home_dir = Path.home()
    config_dir = home_dir / ".evidence_base_classifier"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "config.yaml"

def load_config():
    """Load configuration from file."""
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            st.error(f"Error loading config: {e}")
    return {}

def save_config(config):
    """Save configuration to file."""
    config_path = get_config_path()
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        st.error(f"Error saving config: {e}")
        return False

# Load existing config
if 'config_loaded' not in st.session_state:
    st.session_state.config = load_config()
    st.session_state.config_loaded = True

st.title("Evidence Base Classifier")

# --- Sidebar for settings ---
with st.sidebar:
    st.header("API Keys")
    
    # Get default values from config
    default_openai_key = st.session_state.config.get("openai_api_key", "")
    default_anthropic_key = st.session_state.config.get("anthropic_api_key", "")
    default_model = st.session_state.config.get("model_selection", "o3")
    
    openai_api_key = st.text_input("OpenAI API Key", 
                                   value=default_openai_key if default_openai_key else "",
                                   type="password", 
                                   key="openai_api_key",
                                   help="Get your OpenAI API key from: https://platform.openai.com/api-keys (starts with 'sk-' or 'sk-proj-')")
    anthropic_api_key = st.text_input("Anthropic API Key", 
                                      value=default_anthropic_key if default_anthropic_key else "",
                                      type="password", 
                                      key="anthropic_api_key",
                                      help="Get your Anthropic API key from: https://console.anthropic.com/ (starts with 'sk-ant-')")

    st.header("Model Selection")
    model_options = ["o3", "claude-opus-4"]
    default_index = model_options.index(default_model) if default_model in model_options else 0
    model_selection = st.selectbox(
        "Select Model",
        model_options,
        index=default_index,
        help="Claude Opus 4 uses streaming for better handling of large documents"
    )
    
    st.header("Configuration")
    save_config_enabled = st.checkbox("Save API keys and settings to config file", 
                                     value=st.session_state.config.get("save_enabled", False))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Configuration"):
            if save_config_enabled:
                config_to_save = {
                    "openai_api_key": openai_api_key,
                    "anthropic_api_key": anthropic_api_key,
                    "model_selection": model_selection,
                    "save_enabled": save_config_enabled
                }
                if save_config(config_to_save):
                    st.session_state.config = config_to_save
                    st.success("Configuration saved successfully!")
                else:
                    st.error("Failed to save configuration.")
            else:
                # Clear saved config if user disabled saving
                if save_config({"save_enabled": False}):
                    st.session_state.config = {"save_enabled": False}
                    st.success("Configuration cleared!")
    
    with col2:
        if st.button("🗑️ Clear Saved Config", help="Delete saved configuration file and reset to defaults"):
            try:
                config_path = get_config_path()
                if config_path.exists():
                    config_path.unlink()  # Delete the file
                    st.session_state.config = {}
                    st.success("✅ Saved configuration cleared! Please refresh the page.")
                    st.info("🔄 Refresh the page to see the changes take effect.")
                else:
                    st.info("No saved configuration found to clear.")
            except Exception as e:
                st.error(f"Failed to clear configuration: {e}")
    
    if save_config_enabled:
        st.info(f"Config will be saved to: {get_config_path()}")
    else:
        st.info("API keys will not be saved (enable checkbox to persist settings)")
    
    # Show current config status
    if st.session_state.config and any(key in st.session_state.config for key in ["openai_api_key", "anthropic_api_key"]):
        with st.expander("📋 Current Saved Configuration", expanded=False):
            st.write("**Saved settings:**")
            if "openai_api_key" in st.session_state.config and st.session_state.config["openai_api_key"]:
                key_preview = st.session_state.config["openai_api_key"][:8] + "..." if len(st.session_state.config["openai_api_key"]) > 8 else "Set"
                st.write(f"• OpenAI API Key: {key_preview}")
            if "anthropic_api_key" in st.session_state.config and st.session_state.config["anthropic_api_key"]:
                key_preview = st.session_state.config["anthropic_api_key"][:10] + "..." if len(st.session_state.config["anthropic_api_key"]) > 10 else "Set"
                st.write(f"• Anthropic API Key: {key_preview}")
            if "model_selection" in st.session_state.config:
                st.write(f"• Model: {st.session_state.config['model_selection']}")
            st.warning("💡 If keys were saved incorrectly, use 'Clear Saved Config' button above")

# Check for conditions that should prevent processing.
is_claude_selected = "claude" in model_selection.lower()
is_openai_selected = "o3" in model_selection.lower()

# Validate API key formats
def validate_api_keys():
    """Validate that API keys match their expected formats."""
    issues = []
    
    if openai_api_key:
        if openai_api_key.startswith('sk-ant-'):
            issues.append("❌ **OpenAI API Key field contains an Anthropic key!** Please check your keys.")
        elif openai_api_key.startswith('sk-proj-'):
            # This is likely a project-specific OpenAI key - should be valid
            pass
        elif not openai_api_key.startswith('sk-'):
            issues.append("⚠️ OpenAI API Key should start with 'sk-'")
    
    if anthropic_api_key:
        if anthropic_api_key.startswith('sk-') and not anthropic_api_key.startswith('sk-ant-'):
            issues.append("❌ **Anthropic API Key field contains an OpenAI key!** Please check your keys.")
        elif not anthropic_api_key.startswith('sk-ant-'):
            issues.append("⚠️ Anthropic API Key should start with 'sk-ant-'")
    
    # Check for potential saved config conflicts
    saved_openai = st.session_state.config.get("openai_api_key", "")
    saved_anthropic = st.session_state.config.get("anthropic_api_key", "")
    
    if saved_openai and saved_openai.startswith('sk-ant-'):
        issues.append("🚨 **Saved OpenAI key is actually an Anthropic key!** Use 'Clear Saved Config' button.")
    
    if saved_anthropic and saved_anthropic.startswith('sk-') and not saved_anthropic.startswith('sk-ant-'):
        issues.append("🚨 **Saved Anthropic key is actually an OpenAI key!** Use 'Clear Saved Config' button.")
    
    return issues

key_validation_issues = validate_api_keys()
is_api_key_missing = (is_openai_selected and not openai_api_key) or (is_claude_selected and not anthropic_api_key)
has_key_format_errors = len(key_validation_issues) > 0

with st.sidebar:
    if is_api_key_missing:
        st.warning("Please enter the appropriate API key for the selected model.")
    
    if has_key_format_errors:
        for issue in key_validation_issues:
            st.error(issue)

# --- Main app logic ---
if 'status' not in st.session_state:
    st.session_state.status = "Idle"

st.header("Status")
st.write(f"**Status:** {st.session_state.status}")

st.header("PDF File Selection")
st.write("**Drag and drop your PDF files below**, or click to browse and select multiple files.")
uploaded_files = st.file_uploader(
    "Select PDF files to analyze", 
    type="pdf", 
    accept_multiple_files=True,
    help="You can select multiple PDF files at once. Each file will be processed individually."
)

if uploaded_files:
    st.write(f"📁 **{len(uploaded_files)} PDF file(s) selected:**")
    for file in uploaded_files:
        st.write(f"• {file.name}")

    # Disable button if API key is missing or has format errors
    processing_disabled = is_api_key_missing or has_key_format_errors
    
    if st.button("Process PDFs", disabled=processing_disabled):
        st.session_state.status = "Processing"
        
        # Initialize processors
        pdf_processor = PDFProcessor()
        llm_client = LLMClient()
        results_manager = ResultsManager(model_name=model_selection)
        
        # Initialize LLM client with API keys
        llm_client.initialize_clients(
            openai_api_key=openai_api_key if openai_api_key else None,
            anthropic_api_key=anthropic_api_key if anthropic_api_key else None
        )
        
        # Ensure output and logs directories exist
        os.makedirs("output", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        total_files = len(uploaded_files)
        st.write(f"🔄 **Processing {total_files} PDF files...**")
        
        # Create containers for real-time updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_container = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.write(f"Processing {uploaded_file.name}... ({i + 1} of {total_files})")
            
            try:
                # Step 1: Check if PDF is readable
                is_readable, error_msg = pdf_processor.is_pdf_readable(uploaded_file, uploaded_file.name)
                
                if not is_readable:
                    st.error(f"❌ {uploaded_file.name}: {error_msg}")
                    results_manager.add_error(uploaded_file.name, error_msg)
                    continue
                
                # Step 2: Extract text with OCR fallback
                extracted_text, extraction_success, error_msg = pdf_processor.extract_text_from_pdf(uploaded_file, uploaded_file.name)
                
                if not extraction_success:
                    st.error(f"❌ {uploaded_file.name}: Text extraction failed - {error_msg}")
                    results_manager.add_error(uploaded_file.name, f"Text extraction failed: {error_msg}")
                    continue
                
                # Step 3: Analyze with LLM
                # Give user feedback about processing time for large documents
                estimated_tokens = len(extracted_text) // 4
                if "claude" in model_selection.lower() and estimated_tokens > 150000:
                    status_text.write(f"⏳ Analyzing large document {uploaded_file.name} with {model_selection}... This may take several minutes. ({i + 1} of {total_files})")
                else:
                    status_text.write(f"Analyzing {uploaded_file.name} with {model_selection}... ({i + 1} of {total_files})")
                
                llm_result, analysis_success, error_msg = llm_client.analyze_paper(
                    extracted_text, uploaded_file.name, model_selection
                )
                
                if not analysis_success:
                    st.error(f"❌ {uploaded_file.name}: LLM analysis failed - {error_msg}")
                    results_manager.add_error(uploaded_file.name, f"LLM analysis failed: {error_msg}")
                    continue
                
                # Step 4: Validate result
                is_valid, validation_error = llm_client.validate_result(llm_result)
                
                if not is_valid:
                    st.error(f"❌ {uploaded_file.name}: Invalid LLM response - {validation_error}")
                    results_manager.add_error(uploaded_file.name, f"Invalid LLM response: {validation_error}")
                    continue
                
                # Step 5: Store successful result
                results_manager.add_result(llm_result, uploaded_file.name)
                
                # Display success with key details
                decision = llm_result.get('inclusion_decision', 'Unknown')
                category = llm_result.get('category', 'Unknown')
                title = llm_result.get('article_title', 'Unknown')
                
                if decision == "Included":
                    st.success(f"✅ {uploaded_file.name}: **INCLUDED** ({category}) - {title}")
                else:
                    st.info(f"ℹ️ {uploaded_file.name}: **EXCLUDED** - {title}")
                
                # Show detailed result in expander
                with st.expander(f"Analysis Details: {uploaded_file.name}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Decision:**", decision)
                        st.write("**Category:**", category)
                        st.write("**Title:**", title)
                    with col2:
                        st.write("**Text Length:**", f"{len(extracted_text):,} characters")
                        st.write("**Model Used:**", model_selection)
                    
                    st.write("**Justification:**")
                    st.write(llm_result.get('justification', 'N/A'))
                    
                    st.write("**Detailed Reasoning:**")
                    st.write(llm_result.get('detailed_reasoning', 'N/A'))
                    
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                st.error(f"❌ {uploaded_file.name}: {error_msg}")
                results_manager.add_error(uploaded_file.name, error_msg)
            
            # Update stats display
            stats = results_manager.get_stats()
            stats_container.write(f"**Progress:** Successful: {stats['successful']}, Failed: {stats['failed']}, Remaining: {total_files - stats['total_processed']}")
        
        # Final status update
        progress_bar.progress(1.0)
        status_text.write("Processing completed!")
        
        # Display comprehensive summary
        st.write(results_manager.get_results_summary())
        
        # Display failed files table if any failures occurred
        if results_manager.errors:
            st.write("### Failed Files")
            failed_files_data = results_manager.get_failed_files_for_display()
            st.dataframe(failed_files_data, use_container_width=True)
        
        # Export results and provide download links
        if results_manager.results:
            try:
                csv_filepath = results_manager.export_to_csv()
                with open(csv_filepath, 'rb') as f:
                    st.download_button(
                        label="📊 Download Results CSV",
                        data=f.read(),
                        file_name=os.path.basename(csv_filepath),
                        mime='text/csv'
                    )
            except Exception as e:
                st.error(f"Error exporting results: {e}")
        
        # Export errors if any
        if results_manager.errors:
            try:
                # CSV export
                error_csv_filepath = results_manager.export_errors_to_csv()
                with open(error_csv_filepath, 'rb') as f:
                    st.download_button(
                        label="📋 Download Error Log CSV",
                        data=f.read(),
                        file_name=os.path.basename(error_csv_filepath),
                        mime='text/csv'
                    )
                
                # Text export
                error_txt_filepath = results_manager.export_errors_to_text()
                with open(error_txt_filepath, 'rb') as f:
                    st.download_button(
                        label="📝 Download Error Log TXT",
                        data=f.read(),
                        file_name=os.path.basename(error_txt_filepath),
                        mime='text/plain'
                    )
            except Exception as e:
                st.error(f"Error exporting error logs: {e}")

        st.session_state.status = "Completed"
else:
    st.info("👆 Please select PDF files to begin processing.")
