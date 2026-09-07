/// MotifAssetRegistry centralizes the mapping between batik motif IDs,
/// model prediction labels (35 batik classes), and curated photographic assets in assets/images/motifs/.
class MotifAssetRegistry {
  MotifAssetRegistry._();

  static const String basePath = 'assets/images/motifs/';

  /// Explicit mapping from motif IDs / raw classification labels to asset image filenames.
  /// Covers all 35 authentic batik classes from the 36-class model (excluding 'non_batik').
  static const Map<String, String> _assetMap = {
    // 0. Aceh_Pintu_Aceh
    'aceh_pintu_aceh': 'batik_pintu_aceh.jpg',
    'pintu_aceh': 'batik_pintu_aceh.jpg',
    'batik-aceh': 'batik_pintu_aceh.jpg',

    // 1. Bali_Barong
    'bali_barong': 'batik_bali_barong.jpg',
    'barong': 'batik_bali_barong.jpg',

    // 2. batik-bali
    'batik-bali': 'batik_bali.jpg',
    'batik_bali': 'batik_bali.jpg',
    'bali': 'batik_bali.jpg',

    // 3. batik-betawi
    'batik-betawi': 'batik_betawi.jpg',
    'batik_betawi': 'batik_betawi.jpg',
    'betawi': 'batik_betawi.jpg',

    // 4. batik-celup
    'batik-celup': 'batik_celup.jpg',
    'batik_celup': 'batik_celup.jpg',
    'celup': 'batik_celup.jpg',
    'jumputan': 'batik_celup.jpg',

    // 5. batik-cendrawasih
    'batik-cendrawasih': 'batik_cendrawasih.jpg',
    'batik_cendrawasih': 'batik_cendrawasih.jpg',
    'cendrawasih': 'batik_cendrawasih.jpg',
    'batik-papua': 'batik_cendrawasih.jpg',

    // 6. batik-ceplok
    'batik-ceplok': 'batik_ceplok.jpg',
    'batik_ceplok': 'batik_ceplok.jpg',
    'ceplok': 'batik_ceplok.jpg',

    // 7. batik-ciamis
    'batik-ciamis': 'batik_ciamis.jpg',
    'batik_ciamis': 'batik_ciamis.jpg',
    'ciamis': 'batik_ciamis.jpg',
    'ciamisan': 'batik_ciamis.jpg',

    // 8. batik-garutan
    'batik-garutan': 'batik_garutan.jpg',
    'batik_garutan': 'batik_garutan.jpg',
    'garutan': 'batik_garutan.jpg',
    'garut': 'batik_garutan.jpg',

    // 9. batik-gentongan
    'batik-gentongan': 'batik_gentongan.jpg',
    'batik_gentongan': 'batik_gentongan.jpg',
    'gentongan': 'batik_gentongan.jpg',

    // 10. batik-kawung
    'batik-kawung': 'batik_kawung.jpg',
    'batik_kawung': 'batik_kawung.jpg',
    'kawung': 'batik_kawung.jpg',

    // 11. batik-keraton
    'batik-keraton': 'batik_keraton.jpg',
    'batik_keraton': 'batik_keraton.jpg',
    'keraton': 'batik_keraton.jpg',

    // 12. batik-lasem
    'batik-lasem': 'batik_lasem.jpg',
    'batik_lasem': 'batik_lasem.jpg',
    'lasem': 'batik_lasem.jpg',
    'lasem_tiga_negeri': 'batik_lasem.jpg',

    // 13. batik-megamendung
    'batik-megamendung': 'batik_megamendung.jpg',
    'batik_megamendung': 'batik_megamendung.jpg',
    'megamendung': 'batik_megamendung.jpg',

    // 14. batik-parang
    'batik-parang': 'batik_parang.jpg',
    'batik_parang': 'batik_parang.jpg',
    'parang': 'batik_parang.jpg',
    'parang_rusak': 'batik_parang.jpg',
    'batik-parang-rusak': 'batik_parang.jpg',
    'parang_barong': 'batik_parang.jpg',

    // 15. batik-pekalongan
    'batik-pekalongan': 'batik_pekalongan.jpg',
    'batik_pekalongan': 'batik_pekalongan.jpg',
    'pekalongan': 'batik_pekalongan.jpg',
    'pekalongan_tujuh_rupa': 'batik_pekalongan.jpg',

    // 16. batik-priangan
    'batik-priangan': 'batik_priangan.jpg',
    'batik_priangan': 'batik_priangan.jpg',
    'priangan': 'batik_priangan.jpg',

    // 17. batik-sekar
    'batik-sekar': 'batik_sekar_jagad.jpg',
    'batik_sekar': 'batik_sekar_jagad.jpg',
    'sekar': 'batik_sekar_jagad.jpg',
    'sekar_jagad': 'batik_sekar_jagad.jpg',
    'batik-sekar-jagad': 'batik_sekar_jagad.jpg',

    // 18. batik-sidoluhur
    'batik-sidoluhur': 'batik_sidoluhur.jpg',
    'batik_sidoluhur': 'batik_sidoluhur.jpg',
    'sidoluhur': 'batik_sidoluhur.jpg',

    // 19. batik-sidomukti
    'batik-sidomukti': 'batik_sidomukti.jpg',
    'batik_sidomukti': 'batik_sidomukti.jpg',
    'sidomukti': 'batik_sidomukti.jpg',

    // 20. batik-sogan
    'batik-sogan': 'batik_sogan.jpg',
    'batik_sogan': 'batik_sogan.jpg',
    'sogan': 'batik_sogan.jpg',

    // 21. batik-tambal
    'batik-tambal': 'batik_tambal.jpg',
    'batik_tambal': 'batik_tambal.jpg',
    'tambal': 'batik_tambal.jpg',

    // 22. DKI_Ondel_Ondel
    'dki_ondel_ondel': 'batik_ondel_ondel.jpg',
    'ondel_ondel': 'batik_ondel_ondel.jpg',
    'ondel-ondel': 'batik_ondel_ondel.jpg',

    // 23. Jawa_Timur_Pring
    'jawa_timur_pring': 'batik_pring_sedapur.jpg',
    'pring_sedapur': 'batik_pring_sedapur.jpg',
    'pring': 'batik_pring_sedapur.jpg',

    // 24. Kalimantan_Dayak
    'kalimantan_dayak': 'batik_dayak.jpg',
    'dayak': 'batik_dayak.jpg',
    'batik-dayak': 'batik_dayak.jpg',

    // 25. Lampung_Gajah
    'lampung_gajah': 'batik_lampung_gajah.jpg',
    'gajah_lampung': 'batik_lampung_gajah.jpg',
    'lampung': 'batik_lampung_gajah.jpg',

    // 26. Madura_Mataketeran
    'madura_mataketeran': 'batik_madura_mataketeran.jpg',
    'mataketeran': 'batik_madura_mataketeran.jpg',
    'madura': 'batik_madura_mataketeran.jpg',

    // 27. Maluku_Pala
    'maluku_pala': 'batik_maluku_pala.jpg',
    'pala_maluku': 'batik_maluku_pala.jpg',
    'pala': 'batik_maluku_pala.jpg',
    'batik-maluku': 'batik_maluku_pala.jpg',

    // 28. NTB_Lumbung
    'ntb_lumbung': 'batik_ntb_lumbung.jpg',
    'lumbung': 'batik_ntb_lumbung.jpg',
    'lumbung_sasak': 'batik_ntb_lumbung.jpg',

    // 29. Papua_Asmat
    'papua_asmat': 'batik_papua_asmat.jpg',
    'asmat': 'batik_papua_asmat.jpg',

    // 30. Papua_Cendrawasih
    'papua_cendrawasih': 'batik_papua_cendrawasih.jpg',

    // 31. Papua_Tifa
    'papua_tifa': 'batik_papua_tifa.jpg',
    'tifa': 'batik_papua_tifa.jpg',

    // 32. Sulawesi_Selatan_Lontara
    'sulawesi_selatan_lontara': 'batik_lontara.jpg',
    'lontara': 'batik_lontara.jpg',
    'sulsel_lontara': 'batik_lontara.jpg',

    // 33. Sumatera_Barat_Rumah_Minang
    'sumatera_barat_rumah_minang': 'batik_minang.jpg',
    'rumah_minang': 'batik_minang.jpg',
    'minang_rangkiang': 'batik_minang.jpg',
    'minang': 'batik_minang.jpg',

    // 34. Sumatera_Utara_Boraspati
    'sumatera_utara_boraspati': 'batik_boraspati.jpg',
    'boraspati': 'batik_boraspati.jpg',
    'boraspati_ni_tano': 'batik_boraspati.jpg',
  };

  /// Returns the asset path for a motif ID or raw label, or null if not registered.
  static String? getAssetPath(String idOrLabel) {
    final key = idOrLabel.toLowerCase().trim().replaceAll(' ', '_');
    if (_assetMap.containsKey(key)) {
      return '$basePath${_assetMap[key]}';
    }

    // Secondary substring lookup
    for (final entry in _assetMap.entries) {
      if (key == entry.key || key.contains(entry.key) || entry.key.contains(key)) {
        return '$basePath${entry.value}';
      }
    }

    return null;
  }

  /// Whether a motif has a verified photographic asset.
  static bool hasAsset(String idOrLabel) {
    return getAssetPath(idOrLabel) != null;
  }

  /// Total count of unique photographic motif assets mapped.
  static int get totalUniqueAssets {
    return _assetMap.values.toSet().length;
  }
}
