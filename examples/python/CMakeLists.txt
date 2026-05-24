 INSTALL(FILES
          Vision_files_preprocess.py
          # Quadra_files_preprocess.py not yet ready to be installed
          projdata_profiles.py
          LBFGSBPC.py
          DESTINATION ${PYTHON_DEST}/stir/)
 INSTALL(DIRECTORY projdata_visualisation  DESTINATION ${PYTHON_DEST}/stir/)
