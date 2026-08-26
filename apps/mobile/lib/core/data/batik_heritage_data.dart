import 'package:flutter/material.dart';

/// BatikHeritageItem holds factual cultural, historical, and regional metadata for a batik motif.
class BatikHeritageItem {
  final String id;
  final List<String> rawLabels;
  final String name;
  final String region;
  final String province;
  final String island;
  final String category; // Keraton, Pesisir, Pedalaman, Kontemporer
  final String shortDescription;
  final String philosophy;
  final String history;
  final String visualCharacteristics;
  final String usageOccasion;
  final Color primaryColor;
  final Color secondaryColor;
  final String tag;

  const BatikHeritageItem({
    required this.id,
    required this.rawLabels,
    required this.name,
    required this.region,
    required this.province,
    required this.island,
    required this.category,
    required this.shortDescription,
    required this.philosophy,
    required this.history,
    required this.visualCharacteristics,
    required this.usageOccasion,
    required this.primaryColor,
    required this.secondaryColor,
    required this.tag,
  });

  bool matches(String query) {
    final q = query.toLowerCase();
    return name.toLowerCase().contains(q) ||
        region.toLowerCase().contains(q) ||
        province.toLowerCase().contains(q) ||
        island.toLowerCase().contains(q) ||
        category.toLowerCase().contains(q) ||
        tag.toLowerCase().contains(q);
  }
}

/// Static educational database of Indonesian batik motifs.
class BatikHeritageData {
  static const List<BatikHeritageItem> allMotifs = [
    BatikHeritageItem(
      id: 'parang_rusak',
      rawLabels: ['batik-parang', 'batik-parang-rusak', 'parang', 'Parang_Barong'],
      name: 'Parang Rusak',
      region: 'Yogyakarta & Surakarta',
      province: 'D.I. Yogyakarta & Jawa Tengah',
      island: 'Jawa',
      category: 'Keraton (Larangan)',
      shortDescription: 'Simbol ketabahan manusia dalam mengendalikan hawa nafsu dan perjuangan tanpa henti.',
      philosophy:
          'Bentuk geometris miring menyerupai ombak karang laut selatan melambangkan semangat pantang menyerah, keteguhan hati, kesinambungan budi pekerti luhur, dan kewibawaan kepemimpinan.',
      history:
          'Diciptakan oleh Sultan Agung Hanyokrokusumo saat bertapa di pesisir Pantai Selatan (Parangtritis). Dahulu merupakan motif larangan (Batik Larangan) yang hanya boleh dikenakan oleh raja dan bangsawan keraton.',
      visualCharacteristics:
          'Pola garis diagonal miring berulang (lereng) yang membentuk jajaran huruf "S" sambung-menyambung dengan aksen mlinjon di sela garis.',
      usageOccasion: 'Upacara adat keraton, prosesi sakral, dan pakaian kebesaran raja/bangsawan.',
      primaryColor: Color(0xFF5A2A18),
      secondaryColor: Color(0xFFC8963E),
      tag: 'Batik Larangan',
    ),
    BatikHeritageItem(
      id: 'megamendung',
      rawLabels: ['batik-megamendung', 'megamendung'],
      name: 'Megamendung',
      region: 'Cirebon',
      province: 'Jawa Barat',
      island: 'Jawa',
      category: 'Pesisir (Akulturasi)',
      shortDescription: 'Visualisasi awan pembawa hujan penyejuk kehidupan berpadu dengan akulturasi Tionghoa.',
      philosophy:
          'Awan mendung yang menaungi bumi bermakna kesabaran, kepala dingin, dan ketenangan jiwa dalam menghadapi segala permasalahan hidup agar membawa kesuburan dan keberkahan bagi sekitar.',
      history:
          'Lahir dari akulturasi budaya keraton Cirebon dengan tradisi Tionghoa ketika Sunan Gunung Jati menikahi Putri Ong Tien dari Tiongkok pada abad ke-16.',
      visualCharacteristics:
          'Garis lengkung awan horizontal dengan gradasi warna 7 tingkatan (gradasi lapis saphire/biru atau merah) yang dinamis meruncing.',
      usageOccasion: 'Busana resmi, upacara adat pesisir, perayaan seni, dan busana kontemporer.',
      primaryColor: Color(0xFF1D3557),
      secondaryColor: Color(0xFF457B9D),
      tag: 'Warisan Cirebon',
    ),
    BatikHeritageItem(
      id: 'kawung',
      rawLabels: ['batik-kawung', 'kawung'],
      name: 'Kawung',
      region: 'Yogyakarta & Surakarta',
      province: 'D.I. Yogyakarta & Jawa Tengah',
      island: 'Jawa',
      category: 'Keraton (Geometris)',
      shortDescription: 'Simbol kesucian hati, keadilan, serta asal-usul kehidupan manusia yang seimbang.',
      philosophy:
          'Bentuk lonjong empat penjuru mata angin bermakna konsep sedulur papat lima pancer, melambangkan manusia yang mampu mengendalikan empat nafsu untuk mencapai kesucian batin dan keadilan moral.',
      history:
          'Merupakan salah satu motif tertua di Nusantara yang telah terdokumentasi sejak abad ke-13 pada relief candi Jawa Kuno (Candi Prambanan dan Candi Panataran).',
      visualCharacteristics:
          'Bentuk elips menyerupai buah kolang-kaling atau daun teratai yang tersusun simetris bersilangan membentuk pola lingkaran harmonis.',
      usageOccasion: 'Upacara kenegaraan, prosesi pernikahan adat Jawa, dan peribadatan formal.',
      primaryColor: Color(0xFF6B3E26),
      secondaryColor: Color(0xFFD4A373),
      tag: 'Klasik Mataram',
    ),
    BatikHeritageItem(
      id: 'bali_barong',
      rawLabels: ['Bali_Barong', 'batik-bali', 'bali'],
      name: 'Bali Barong',
      region: 'Denpasar & Gianyar',
      province: 'Bali',
      island: 'Bali',
      category: 'Pesisir & Sakral',
      shortDescription: 'Epitom pelindung spiritual dan simbol kemenangan kebajikan (Dharma) melawan kejahatan.',
      philosophy:
          'Figur Barong sebagai simbol Dharma merepresentasikan kekuatan magis pelindung manusia dan penyeimbang keharmonisan alam semesta (Tri Hita Karana).',
      history:
          'Berkembang di pusat seni Gianyar dan Denpasar, memadukan teknik canting Jawa dengan kekayaan ornamen ukiran pura dan mitologi Hindu Bali.',
      visualCharacteristics:
          'Figur kepala Barong megah dengan taring dan ornamen pepatran (sulur daun kamboja, teratai) dengan warna cerah dan kontras ekspresif.',
      usageOccasion: 'Upacara keagamaan Hindu Bali, festival seni budaya, dan busana pesta adat.',
      primaryColor: Color(0xFF8B1E1E),
      secondaryColor: Color(0xFFDDA15E),
      tag: 'Khas Pulau Dewata',
    ),
    BatikHeritageItem(
      id: 'sekar_jagad',
      rawLabels: ['batik-sekar', 'batik-sekar-jagad', 'sekar_jagad'],
      name: 'Sekar Jagad',
      region: 'Yogyakarta & Solo',
      province: 'D.I. Yogyakarta & Jawa Tengah',
      island: 'Jawa',
      category: 'Keraton (Botanikal)',
      shortDescription: 'Bunga keindahan dunia yang merangkum keanekaragaman flora nusantara yang memikat.',
      philosophy:
          'Nama "Sekar Jagad" berasal dari kata "kar" (peta/keindahan) dan "jagad" (dunia). Melambangkan keindahan keberagaman suku bangsa dan budaya dalam satu kesatuan yang harmonis.',
      history:
          'Muncul sejak masa Kesultanan Mataram Islam abad ke-18 sebagai manifestasi rasa syukur terhadap kemakmuran bumi Nusantara.',
      visualCharacteristics:
          'Pola pulau-pulau kecil asimetris (peta) yang di dalamnya memuat beragam isen-isen bunga, dedaunan, dan ornamen batik mini.',
      usageOccasion: 'Busana pernikahan (pakaian orang tua pengantin), wisuda, dan acara kenegaraan.',
      primaryColor: Color(0xFF783D19),
      secondaryColor: Color(0xFFE9C46A),
      tag: 'Simbol Keindahan',
    ),
    BatikHeritageItem(
      id: 'pintu_aceh',
      rawLabels: ['Aceh_Pintu_Aceh', 'pintu_aceh', 'batik-aceh'],
      name: 'Pintu Aceh',
      region: 'Banda Aceh',
      province: 'Aceh',
      island: 'Sumatera',
      category: 'Pesisir (Arsitektural)',
      shortDescription: 'Cerminan kepribadian masyarakat Aceh yang santun, ramah, namun teguh memegang prinsip.',
      philosophy:
          'Mengambil inspirasi dari pintu rumah adat Aceh yang rendah mengharuskan tamu menunduk, bermakna kerendahan hati, adab sopan santun, dan ketakwaan religius.',
      history:
          'Awalnya merupakan motif perhiasan logam emas peninggalan Sultan Iskandar Muda yang kemudian diadaptasikan ke seni wastra batik modern Aceh pada dekade 1970-an.',
      visualCharacteristics:
          'Siluet gerbang pintu ramping dengan ornamen sulur geometris teratur yang rapat dan bernuansa islami tanpa figur makhluk bernyawa.',
      usageOccasion: 'Pesta adat Meukuta, resepsi pernikahan formal, dan busana kerja kelembagaan.',
      primaryColor: Color(0xFF2A5235),
      secondaryColor: Color(0xFFE0A96D),
      tag: 'Serambi Mekkah',
    ),
    BatikHeritageItem(
      id: 'kalimantan_dayak',
      rawLabels: ['Kalimantan_Dayak', 'dayak', 'batik-dayak'],
      name: 'Dayak Batang Garing',
      region: 'Palangka Raya & Samarinda',
      province: 'Kalimantan Tengah & Timur',
      island: 'Kalimantan',
      category: 'Pedalaman & Kosmologis',
      shortDescription: 'Pohon kehidupan penopang jagat raya dan hubungan sakral manusia dengan alam leluhur.',
      philosophy:
          'Pohon Batang Garing melambangkan tiga alam (dunia atas, dunia tengah tempat manusia berakar, dan dunia bawah), mengajarkan kelestarian ekosistem hutan dan penghormatan leluhur.',
      history:
          'Diwariskan secara turun-temurun oleh suku Dayak Ngaju dan Kenyah, diabadikan dalam kain tenun dan seni lukis tameng perang sebelum diadaptasikan menjadi seni batik.',
      visualCharacteristics:
          'Lengkungan sulur meliuk dinamis (kaligrafi Dayak), motif tameng perisai (Talawang), dan burung Enggang gading dengan warna merah, kuning, dan hitam pekat.',
      usageOccasion: 'Ritual adat Tiwah, festival Erau, upacara perkawinan adat, dan perayaan panen.',
      primaryColor: Color(0xFF9E2A2B),
      secondaryColor: Color(0xFFF4A261),
      tag: 'Kearifan Borneo',
    ),
    BatikHeritageItem(
      id: 'papua_cendrawasih',
      rawLabels: ['Papua_Cendrawasih', 'Papua_Tifa', 'cendrawasih', 'batik-papua'],
      name: 'Papua Cendrawasih & Tifa',
      region: 'Jayapura & Asmat',
      province: 'Papua',
      island: 'Papua',
      category: 'Kontemporer Etnik',
      shortDescription: 'Pesona burung surga dan instrumen magis pengobar semangat persaudaraan bumi cenderawasih.',
      philosophy:
          'Burung Cendrawasih melambangkan keanggunan, keelokan alami, dan kedamaian, sedangkan Tifa melambangkan ketegasan pemimpin dalam memanggil persatuan warga.',
      history:
          'Diperkenalkan dan dikembangkan sejak tahun 1985 melalui program pertukaran perajin batik Solo ke Papua, melahirkan sintesis seni canting dengan ornamen ukir Asmat dan Sentani.',
      visualCharacteristics:
          'Figur burung berekor panjang melayang anggun dipadu ornamen tifa berkulit biawak, ukiran geometris spiral Asmat, dan warna alam tanah terakota-emas cerah.',
      usageOccasion: 'Festival Lembah Baliem, upacara kedatangan tamu kehormatan, dan busana formal Nusantara.',
      primaryColor: Color(0xFF5E2B6D),
      secondaryColor: Color(0xFFE76F51),
      tag: 'Permata Timur',
    ),
    BatikHeritageItem(
      id: 'pekalongan_tujuh_rupa',
      rawLabels: ['batik-pekalongan', 'pekalongan'],
      name: 'Pekalongan Tujuh Rupa',
      region: 'Pekalongan',
      province: 'Jawa Tengah',
      island: 'Jawa',
      category: 'Pesisir (Multikultural)',
      shortDescription: 'Ledakan warna warni pesisir yang dinamis mencerminkan keterbukaan dan kebebasan berekspresi.',
      philosophy:
          'Keanekaragaman 7 ragam flora dan fauna dalam satu helai kain melambangkan keluwesan beradaptasi, keberanian berinovasi, dan harmoni pergaulan antarbangsa.',
      history:
          'Pekalongan sebagai pelabuhan niaga strategis menyerap budaya Arab, Cina, India, dan Belanda (Batik Encim & Batik Belanda) yang melahirkan ragam hias paling ekspresif di Nusantara.',
      visualCharacteristics:
          'Warna-warna cerah ceria (merah muda, hijau pupus, toska), gambar buketan bunga Eropa, burung merak, dan kupu-kupu dengan isen cecek yang sangat halus.',
      usageOccasion: 'Busana pesta santai, resepsi pernikahan modern, dan kain sarung harian.',
      primaryColor: Color(0xFF264653),
      secondaryColor: Color(0xFF2A9D8F),
      tag: 'Kota Batik Dunia',
    ),
    BatikHeritageItem(
      id: 'lasem_tiga_negeri',
      rawLabels: ['batik-lasem', 'lasem'],
      name: 'Lasem Tiga Negeri',
      region: 'Rembang',
      province: 'Jawa Tengah',
      island: 'Jawa',
      category: 'Pesisir (Akulturasi Sejati)',
      shortDescription: 'Mahakarya perpaduan warna merah darah ayam Tionghoa, biru pekalongan, dan cokelat soga keraton.',
      philosophy:
          'Tiga warna pokok melambangkan toleransi, persaudaraan lintas etnis Tionghoa dan Jawa, serta keselarasan hidup dalam keragaman budaya.',
      history:
          'Proses pewarnaan legendaris yang dahulu dikerjakan di tiga kota berbeda: Merah di Lasem, Biru di Pekalongan/Kudus, dan Cokelat Soga di Solo/Yogyakarta.',
      visualCharacteristics:
          'Motif burung Hong (Phoenix), naga (Liong), dan bunga teratai berpadu dengan motif parang dan sekar jagad Jawa yang klasik.',
      usageOccasion: 'Pernikahan peranakan Tionghoa-Jawa, perayaan Imlek, dan koleksi adibusana bernilai tinggi.',
      primaryColor: Color(0xFF9E0031),
      secondaryColor: Color(0xFFC8963E),
      tag: 'Pusaka Tiga Negeri',
    ),
    BatikHeritageItem(
      id: 'minang_rangkiang',
      rawLabels: ['Sumatera_Barat_Rumah_Minang', 'rumah_minang', 'minang'],
      name: 'Minangkabau Rangkiang',
      region: 'Padang & Bukittinggi',
      province: 'Sumatera Barat',
      island: 'Sumatera',
      category: 'Pesisir (Filosofi Adat)',
      shortDescription: 'Simbol lumbung pangan keluarga, kesejahteraan masyarakat, dan ketahanan kaum.',
      philosophy:
          'Rangkiang sebagai tempat penyimpanan padi merepresentasikan kearifan hidup berhemat, kesiapan menghadapi masa sulit, dan kemakmuran bersama yang berkeadilan.',
      history:
          'Terinspirasi dari ukiran kayu dinding Rumah Gadang (Itiak Pulang Patang, Kaluak Paku) yang diintegrasikan ke tradisi batik tanah liek Sumatera Barat.',
      visualCharacteristics:
          'Siluet atap gonjong rumah gadang bertingkat dipadu ukiran kaluak paku melingkar simetris dengan pewarnaan alami tanah liat hangat.',
      usageOccasion: 'Upacara Batagak Gala, alek nagari, dan pakaian formal adat Minang.',
      primaryColor: Color(0xFF7A4419),
      secondaryColor: Color(0xFFE29578),
      tag: 'Kearifan Minang',
    ),
    BatikHeritageItem(
      id: 'pala_maluku',
      rawLabels: ['Maluku_Pala', 'pala', 'batik-maluku'],
      name: 'Maluku Buah Pala',
      region: 'Banda & Ambon',
      province: 'Maluku',
      island: 'Maluku',
      category: 'Kepulauan Rempah',
      shortDescription: 'Emas hitam aromatik yang mengubah sejarah perdagangan rempah jalur maritim dunia.',
      philosophy:
          'Buah pala melambangkan kekayaan alam yang bernilai tinggi, ketahanan jati diri rempah nusantara, dan keharuman budi pekerti warga Maluku.',
      history:
          'Kepulauan Banda adalah satu-satunya produsen pala asli dunia pada era kolonial. Motif ini merayakan warisan agung Kepulauan Rempah (Spice Islands).',
      visualCharacteristics:
          'Buah pala merekah memperlihatkan fuli merah menyala berpadu cengkih dan gelombang samudra biru kepulauan.',
      usageOccasion: 'Pesta adat Panas Pela, pesta rakyat Maluku, dan busana daerah formal.',
      primaryColor: Color(0xFF3D2645),
      secondaryColor: Color(0xFFE07A5F),
      tag: 'Warisan Rempah',
    ),
  ];

  /// The featured Motif of the Day
  static const BatikHeritageItem motifOfTheDay = BatikHeritageItem(
    id: 'parang_rusak',
    rawLabels: ['batik-parang', 'batik-parang-rusak', 'parang'],
    name: 'Parang Rusak Barong',
    region: 'Yogyakarta & Surakarta',
    province: 'D.I. Yogyakarta',
    island: 'Jawa',
    category: 'Keraton Mataram',
    shortDescription: 'Simbol ksatria dalam mengendalikan hawa nafsu dan kesinambungan budi pekerti luhur.',
    philosophy:
          'Pola lereng diagonal yang membentuk jajaran huruf "S" tak terputus melambangkan kesinambungan garis keturunan, kepemimpinan bijaksana, dan keberanian melawan godaan batin.',
    history:
          'Diciptakan oleh Sultan Agung Hanyokrokusumo saat bermeditasi di pesisir Parangtritis. Merupakan salah satu pola agung tertua peradaban Mataram.',
    visualCharacteristics:
          'Jajaran motif parang berukuran besar (>8 cm) dengan aksen mlinjon tajam dan warna soga sogan cokelat tanah.',
    usageOccasion: 'Busana pusaka upacara keraton dan prosesi pernikahan agung.',
    primaryColor: Color(0xFF5A2A18),
    secondaryColor: Color(0xFFC8963E),
    tag: 'Batik of the Day',
  );

  /// Trending cultural motifs for the dashboard
  static List<BatikHeritageItem> get trendingMotifs => [
        allMotifs[1], // Megamendung
        allMotifs[2], // Kawung
        allMotifs[3], // Bali Barong
        allMotifs[5], // Pintu Aceh
        allMotifs[6], // Dayak
        allMotifs[7], // Papua Cendrawasih
      ];

  /// Resolves any model prediction raw label to a rich cultural heritage object.
  static BatikHeritageItem lookup(String rawLabel) {
    final cleaned = rawLabel.toLowerCase().replaceAll('-', '_');
    for (final item in allMotifs) {
      for (final l in item.rawLabels) {
        final cl = l.toLowerCase().replaceAll('-', '_');
        if (cleaned.contains(cl) || cl.contains(cleaned)) {
          return item;
        }
      }
    }
    // Fallback: Generate a clean placeholder item
    final formattedName = rawLabel.replaceAll('-', ' ').replaceAll('_', ' ');
    return BatikHeritageItem(
      id: cleaned,
      rawLabels: [rawLabel],
      name: formattedName,
      region: 'Nusantara',
      province: 'Indonesia',
      island: 'Indonesia',
      category: 'Tradisional',
      shortDescription: 'Ragam hias warisan budaya wastra nusantara dengan nilai filosofis mendalam.',
      philosophy:
          'Setiap guratan canting dan malam pada motif wastra nusantara menyimpan doa, harapan kebajikan, dan perwujudan kearifan lokal leluhur bangsa.',
      history:
          'Batik telah diakui oleh UNESCO sebagai Warisan Kemanusiaan untuk Budaya Lisan dan Nonbendawi (Masterpiece of the Oral and Intangible Heritage of Humanity) sejak 2 Oktober 2009.',
      visualCharacteristics:
          'Karakteristik visual ornamen khas nusantara dengan harmoni komposisi garis, isen-isen tradisional, dan pewarnaan khas daerah.',
      usageOccasion: 'Acara kebudayaan, pertemuan formal, dan busana harian Nusantara.',
      primaryColor: const Color(0xFF8D4925),
      secondaryColor: const Color(0xFFC8963E),
      tag: 'Warisan Nusantara',
    );
  }
}
